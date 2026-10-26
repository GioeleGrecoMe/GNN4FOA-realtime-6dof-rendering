// GNN4FOA v2
// Authors: Gioele Greco, Filippo Marri, Mirco Pezzoli, Fabio Antonacci
// Image and Sound Processing Lab (ISPL), Politecnico di Milano
// Copyright (c) 2026 Gioele Greco, Filippo Marri, Mirco Pezzoli, Fabio Antonacci

#include "PluginProcessor.h"
#include "PluginEditor.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdint>
#include <cstdio>



namespace
{
constexpr const char* kStateVersionProperty = "SPATIAL_STATE_VERSION";
constexpr int kSpatialStateVersion = 9;
constexpr int kFifoCapacityHostFrames = 131072;
constexpr double kModelDeadlineMs = 1000.0 * GNNReconstructorAudioProcessor::kModelBlockSize
                                  / GNNReconstructorAudioProcessor::kModelSampleRate;

void migrateNodeCountParameter(juce::ValueTree tree, int savedSpatialVersion)
{
    // v2 exposes ten manual choices (N=1..10). Preserve old explicit N values
    // when importing a state from the earlier 5-node builds. Legacy Auto maps
    // to N=5; no host-derived cardinality is reintroduced.
    auto migrateNormalisedValue = [savedSpatialVersion] (float oldNorm) noexcept
    {
        oldNorm = juce::jlimit(0.0f, 1.0f, oldNorm);
        int nodeCount = 5;
        if (savedSpatialVersion < 8)
        {
            const int oldChoice = juce::jlimit(0, 5, juce::roundToInt(oldNorm * 5.0f));
            nodeCount = oldChoice == 0 ? 5 : oldChoice;
        }
        else
        {
            const int oldIndex = juce::jlimit(0, 4, juce::roundToInt(oldNorm * 4.0f));
            nodeCount = oldIndex + 1;
        }
        return (float) (nodeCount - 1) / 9.0f;
    };

    if (tree.hasProperty("id") && tree.getProperty("id").toString() == "NODE_COUNT"
        && tree.hasProperty("value"))
        tree.setProperty("value", migrateNormalisedValue((float) tree.getProperty("value")), nullptr);
    else if (tree.hasProperty("NODE_COUNT"))
        tree.setProperty("NODE_COUNT", migrateNormalisedValue((float) tree.getProperty("NODE_COUNT")), nullptr);

    for (int i = 0; i < tree.getNumChildren(); ++i)
        migrateNodeCountParameter(tree.getChild(i), savedSpatialVersion);
}
}

//==============================================================================
// Lock-free interleaved SPSC FIFO
void GNNReconstructorAudioProcessor::InterleavedSpscFifo::prepare(int channelsToUse,
                                                                  int capacityFramesToUse)
{
    channels = channelsToUse;
    capacityFrames = capacityFramesToUse;
    data.assign((size_t) channels * (size_t) capacityFrames, 0.0f);
    reset();
}

void GNNReconstructorAudioProcessor::InterleavedSpscFifo::reset() noexcept
{
    writeCounter.store(0, std::memory_order_relaxed);
    readCounter.store(0, std::memory_order_relaxed);
    std::fill(data.begin(), data.end(), 0.0f);
}

bool GNNReconstructorAudioProcessor::InterleavedSpscFifo::pushFrame(const float* frame) noexcept
{
    const uint64_t w = writeCounter.load(std::memory_order_relaxed);
    const uint64_t r = readCounter.load(std::memory_order_acquire);
    if (w - r >= (uint64_t) capacityFrames)
        return false;

    const size_t base = (size_t) (w % (uint64_t) capacityFrames) * (size_t) channels;
    for (int c = 0; c < channels; ++c)
        data[base + (size_t) c] = frame[c];
    writeCounter.store(w + 1, std::memory_order_release);
    return true;
}

bool GNNReconstructorAudioProcessor::InterleavedSpscFifo::popFrame(float* frame) noexcept
{
    const uint64_t r = readCounter.load(std::memory_order_relaxed);
    const uint64_t w = writeCounter.load(std::memory_order_acquire);
    if (r == w)
        return false;

    const size_t base = (size_t) (r % (uint64_t) capacityFrames) * (size_t) channels;
    for (int c = 0; c < channels; ++c)
        frame[c] = data[base + (size_t) c];
    readCounter.store(r + 1, std::memory_order_release);
    return true;
}

int GNNReconstructorAudioProcessor::InterleavedSpscFifo::discardOldestFrames(int maxFrames) noexcept
{
    if (maxFrames <= 0)
        return 0;

    // This method is called only by the FIFO consumer. Advancing the read
    // counter is therefore an O(1) real-time-safe way of dropping stale
    // frames; the producer observes the updated counter with acquire
    // semantics before reusing those slots.
    const uint64_t r = readCounter.load(std::memory_order_relaxed);
    const uint64_t w = writeCounter.load(std::memory_order_acquire);
    const uint64_t available = w - r;
    const uint64_t toDrop = std::min<uint64_t>(available, (uint64_t) maxFrames);
    if (toDrop == 0)
        return 0;

    readCounter.store(r + toDrop, std::memory_order_release);
    return (int) toDrop;
}

int GNNReconstructorAudioProcessor::InterleavedSpscFifo::getAvailableFrames() const noexcept
{
    const uint64_t r = readCounter.load(std::memory_order_acquire);
    const uint64_t w = writeCounter.load(std::memory_order_acquire);
    return (int) std::min<uint64_t>(w - r, (uint64_t) std::numeric_limits<int>::max());
}

//==============================================================================
// Fixed-cost causal polyphase windowed-sinc sample-rate converter. The full
// coefficient table is built in prepareToPlay(); steady-state processing uses
// only multiply-adds and table lookups. This is important on older Intel Macs:
// Precomputed kernels avoid evaluating sin/cos for every tap in processBlock.
void GNNReconstructorAudioProcessor::StreamingSincResampler::prepare(double sourceRateToUse,
                                                                     double destinationRateToUse,
                                                                     int channelsToUse) noexcept
{
    sourceRate = std::max(1.0, sourceRateToUse);
    destinationRate = std::max(1.0, destinationRateToUse);
    channels = juce::jlimit(1, kMaxChannels, channelsToUse);
    sourceStepPerOutput = sourceRate / destinationRate;
    cutoff = 0.475 * std::min(1.0, destinationRate / sourceRate);
    bypass = std::abs(sourceRate - destinationRate) < 1.0e-9;

    const int half = kTaps / 2;
    for (int phase = 0; phase < kPhases; ++phase)
    {
        const double frac = (double) phase / (double) (kPhases - 1);
        double sumW = 0.0;
        for (int ti = 0; ti < kTaps; ++ti)
        {
            const int tap = ti - half + 1;
            const double t = (double) tap - frac;
            const double sinc = (std::abs(t) < 1.0e-12)
                              ? (2.0 * cutoff)
                              : std::sin(2.0 * juce::MathConstants<double>::pi * cutoff * t)
                                / (juce::MathConstants<double>::pi * t);
            const double window = 0.42
                                + 0.50 * std::cos(juce::MathConstants<double>::pi * t / (double) half)
                                + 0.08 * std::cos(2.0 * juce::MathConstants<double>::pi * t / (double) half);
            const double w = (std::abs(t) <= (double) half) ? sinc * window : 0.0;
            phaseTable[(size_t) phase * kTaps + (size_t) ti] = (float) w;
            sumW += w;
        }
        if (std::abs(sumW) > 1.0e-12)
            for (int ti = 0; ti < kTaps; ++ti)
                phaseTable[(size_t) phase * kTaps + (size_t) ti] /= (float) sumW;
    }
    reset();
}

void GNNReconstructorAudioProcessor::StreamingSincResampler::reset() noexcept
{
    history.fill(0.0f);
    sourceIndex = -1;
    nextOutputSourceTime = 0.0;
}

double GNNReconstructorAudioProcessor::StreamingSincResampler::getDelaySeconds() const noexcept
{
    return bypass ? 0.0 : ((0.5 * (double) kTaps) / sourceRate);
}

float GNNReconstructorAudioProcessor::StreamingSincResampler::getSample(int channel,
                                                                        int64_t absoluteIndex) const noexcept
{
    if (absoluteIndex < 0 || absoluteIndex > sourceIndex || sourceIndex - absoluteIndex >= kHistorySize)
        return 0.0f;
    const size_t slot = (size_t) (absoluteIndex % kHistorySize);
    return history[(size_t) channel * kHistorySize + slot];
}

float GNNReconstructorAudioProcessor::StreamingSincResampler::interpolate(int channel,
                                                                          double delayedTime) const noexcept
{
    if (delayedTime < 0.0)
        return 0.0f;

    const int half = kTaps / 2;
    const int64_t centre = (int64_t) std::floor(delayedTime);
    const double frac = delayedTime - (double) centre;
    const int phase = juce::jlimit(0, kPhases - 1,
                                  juce::roundToInt(frac * (double) (kPhases - 1)));
    const float* coeff = phaseTable.data() + (size_t) phase * kTaps;

    float sum = 0.0f;
    for (int ti = 0; ti < kTaps; ++ti)
    {
        const int tap = ti - half + 1;
        sum += coeff[ti] * getSample(channel, centre + tap);
    }
    return sum;
}

int GNNReconstructorAudioProcessor::StreamingSincResampler::processFrame(
    const float* input,
    float* outputInterleaved,
    int maxOutputFrames) noexcept
{
    ++sourceIndex;
    const size_t slot = (size_t) (sourceIndex % kHistorySize);
    for (int c = 0; c < channels; ++c)
        history[(size_t) c * kHistorySize + slot] = input[c];

    if (bypass)
    {
        if (maxOutputFrames <= 0)
            return 0;
        for (int c = 0; c < channels; ++c)
            outputInterleaved[c] = input[c];
        return 1;
    }

    int produced = 0;
    while (nextOutputSourceTime <= (double) sourceIndex + 1.0e-12 && produced < maxOutputFrames)
    {
        const double delayedTime = nextOutputSourceTime - 0.5 * (double) kTaps;
        for (int c = 0; c < channels; ++c)
            outputInterleaved[(size_t) produced * (size_t) channels + (size_t) c] = interpolate(c, delayedTime);
        ++produced;
        nextOutputSourceTime += sourceStepPerOutput;
    }
    return produced;
}

//==============================================================================
void GNNReconstructorAudioProcessor::ModelWorker::run()
{
    owner.runModelWorker();
}

GNNReconstructorAudioProcessor::GNNReconstructorAudioProcessor()
#ifndef JucePlugin_PreferredChannelConfigurations
    : AudioProcessor(BusesProperties()
                    #if ! JucePlugin_IsMidiEffect
                     #if ! JucePlugin_IsSynth
                      // One flat 40-channel discrete input bus and one 4-channel output.
                      // The user selects N=1..10 in the plug-in; DSP consumes only
                      // the first 4*N channels. No host channel-count detection or
                      // REAPER pin remapping is performed during playback.
                      .withInput("FOA Network Input", juce::AudioChannelSet::discreteChannels(40), true)
                     #endif
                      .withOutput("Reconstructed FOA", juce::AudioChannelSet::discreteChannels(4), true)
                    #endif
                      ),
      apvts(*this, nullptr, "Parameters", createParameters())
#else
    : apvts(*this, nullptr, "Parameters", createParameters())
#endif
{
    micArrays.reserve(kNumMics);
    micArrays.push_back({ 0, -5.0f,  5.0f, 0.0f, 0.0f, false, "Array 1" });
    micArrays.push_back({ 1,  5.0f,  5.0f, 0.0f, 0.0f, false, "Array 2" });
    micArrays.push_back({ 2, -5.0f, -5.0f, 0.0f, 0.0f, false, "Array 3" });
    micArrays.push_back({ 3,  5.0f, -5.0f, 0.0f, 0.0f, false, "Array 4" });
    micArrays.push_back({ 4,  0.0f,  7.0f, 0.0f, 0.0f, false, "Array 5" });
    micArrays.push_back({ 5,  7.0f,  0.0f, 0.0f, 0.0f, false, "Array 6" });
    micArrays.push_back({ 6, -7.0f,  0.0f, 0.0f, 0.0f, false, "Array 7" });
    micArrays.push_back({ 7,  0.0f, -7.0f, 0.0f, 0.0f, false, "Array 8" });
    micArrays.push_back({ 8,  7.0f,  7.0f, 0.0f, 0.0f, false, "Array 9" });
    micArrays.push_back({ 9, -7.0f, -7.0f, 0.0f, 0.0f, false, "Array 10" });
    listener = { 99, 0.0f, 0.0f, 0.0f, 0.0f, true, "Listener" };

    interpModeParam = apvts.getRawParameterValue("INTERP_MODE");
    energyThresholdParam = apvts.getRawParameterValue("ENG_THRESH");
    diracSmoothingParam = apvts.getRawParameterValue("DIRAC_SMOOTH");
    temporalWindowParam = apvts.getRawParameterValue("GNN_WINDOW");
    nodeCountParam = apvts.getRawParameterValue("NODE_COUNT");

    publishSpatialStateFromGui();
}

GNNReconstructorAudioProcessor::~GNNReconstructorAudioProcessor()
{
    releaseResources();
}

const juce::String GNNReconstructorAudioProcessor::getName() const { return JucePlugin_Name; }
bool GNNReconstructorAudioProcessor::acceptsMidi() const { return false; }
bool GNNReconstructorAudioProcessor::producesMidi() const { return false; }
bool GNNReconstructorAudioProcessor::isMidiEffect() const { return false; }
double GNNReconstructorAudioProcessor::getTailLengthSeconds() const { return 0.0; }
int GNNReconstructorAudioProcessor::getNumPrograms() { return 1; }
int GNNReconstructorAudioProcessor::getCurrentProgram() { return 0; }
void GNNReconstructorAudioProcessor::setCurrentProgram(int) {}
const juce::String GNNReconstructorAudioProcessor::getProgramName(int) { return {}; }
void GNNReconstructorAudioProcessor::changeProgramName(int, const juce::String&) {}

int GNNReconstructorAudioProcessor::getSelectedMicCount() const noexcept
{
    // NODE_COUNT is a ten-entry AudioParameterChoice; raw value is 0..9.
    // It is the only authority for N. Host track width, bus activity and signal
    // level can never change the graph cardinality behind the user's back.
    const int choice = nodeCountParam != nullptr
                     ? juce::roundToInt(nodeCountParam->load(std::memory_order_relaxed))
                     : 4; // default N=5
    return gnn4foa::ManualFoaPolicy::sanitiseNodeCount(choice + 1);
}

int GNNReconstructorAudioProcessor::getSceneMicCount() const noexcept
{
    return getSelectedMicCount();
}

juce::AudioProcessorValueTreeState::ParameterLayout GNNReconstructorAudioProcessor::createParameters()
{
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;
    params.push_back(std::make_unique<juce::AudioParameterChoice>(
        "INTERP_MODE", "Interpolation Mode", juce::StringArray { "Delaunay Baseline", "GNN Learned" }, 0));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        "ENG_THRESH", "GNN Threshold: below=Delaunay (dB rel.)", -120.0f, 0.0f, -80.0f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        "DIRAC_SMOOTH", "DirAC Covariance Smoothing", 0.0f, 0.9999f, 0.2f));
    params.push_back(std::make_unique<juce::AudioParameterInt>(
        "GNN_WINDOW", "GNN Temporal Weight Average", 1, DiracGNNEngine::kMaxTemporalWindow, 1));
    params.push_back(std::make_unique<juce::AudioParameterChoice>(
        "NODE_COUNT", "Active FOA Nodes (manual)",
        juce::StringArray { "1", "2", "3", "4", "5", "6", "7", "8", "9", "10" }, 4));
    return { params.begin(), params.end() };
}

void GNNReconstructorAudioProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    releaseResources();

    hostSampleRate = std::max(1.0, sampleRate);
    hostMaxBlockSize = std::max(1, samplesPerBlock);
    resamplingActive = std::abs(hostSampleRate - kModelSampleRate) > 1.0e-9;
    // Internal transport is always provisioned for the complete known-node
    // network. The physical host contract is one stable 40-channel bus; manual
    // N only changes how many leading quartets are consumed by the DSP.
    configuredInputChannels = DiracGNNEngine::kNumInputChannels; // 40 = 10 * FOA4
    configuredMicCount = kNumMics;

    // Manual-only cardinality. This value is read from the plug-in parameter and
    // never inferred from the host during the run. The physical bus remains 40
    // channels so no bus renegotiation is necessary when N is changed manually.
    const int startupActiveMicCount = getSelectedMicCount();
    effectiveMicCount.store(startupActiveMicCount, std::memory_order_relaxed);

    hostInputFifo.prepare(configuredInputChannels, kFifoCapacityHostFrames);
    hostOutputFifo.prepare(DiracGNNEngine::kNumOutputChannels, kFifoCapacityHostFrames);

    inputResampler.prepare(hostSampleRate, kModelSampleRate, configuredInputChannels);
    outputResampler.prepare(kModelSampleRate, hostSampleRate, DiracGNNEngine::kNumOutputChannels);

    workerInputBuffer.setSize(configuredInputChannels, kModelBlockSize, false, true, false);
    workerOutputBuffer.setSize(DiracGNNEngine::kNumOutputChannels, kModelBlockSize, false, true, false);
    workerModelInputFill = 0;

    gnnEngine.prepare(kModelSampleRate, kModelBlockSize);

    // Calibrate the learned backend before audio starts using the selected node
    // count and active matrix backend, then reset all analysis and GNN state.
    {
        // Match the floating-point mode used by the model worker so subnormal
        // values do not distort the startup timing estimate.
        juce::ScopedNoDenormals startupNoDenormals;
        std::array<float, kNumMics> bx {}, by {}, bz {};
        for (int m = 0; m < kNumMics; ++m)
        {
            bx[(size_t) m] = micArrays[(size_t) m].x;
            by[(size_t) m] = micArrays[(size_t) m].y;
            bz[(size_t) m] = micArrays[(size_t) m].z;
        }
        const int calibrationMicCount = startupActiveMicCount;
        startupGNNBenchmarkMs = gnnEngine.benchmarkGNNForwardMs(listener.x, listener.y, listener.z,
                                                                 bx, by, bz, calibrationMicCount);
        gnnEngine.reset();
        gnnEngine.seedGNNForwardTimingEstimate(startupGNNBenchmarkMs, calibrationMicCount);
    }

    lastRenderedSample.fill(0.0f);
    starvationGain = 1.0f;
    resetDiagnostics();

    // Continuity-first compute reservoir.  The target 2010-class x86_64 Mac
    // gets a conservative 128-ms floor and up to 512 ms when the measured GNN
    // cost requires it.  The benchmark is intentionally conservative and the
    // entire reservoir is declared to the host as compensable plug-in latency.
   #if defined(__x86_64__) || defined(_M_X64)
    constexpr int baseReservoirHops = 16; // 128 ms
    constexpr int maxReservoirHops  = 64; // 512 ms
   #else
    constexpr int baseReservoirHops = 8;  // 64 ms
    constexpr int maxReservoirHops  = 40; // 320 ms
   #endif
    const double calibratedMs = startupGNNBenchmarkMs > 0.0
                              ? 2.0 * startupGNNBenchmarkMs + 48.0
                              : (double) baseReservoirHops * kModelDeadlineMs;
    const int calibratedHops = juce::jlimit(baseReservoirHops, maxReservoirHops,
        (int) std::ceil(calibratedMs / kModelDeadlineMs));
    const int blockEquivalentModelSamples = (int) std::ceil((double) hostMaxBlockSize * kModelSampleRate / hostSampleRate);
    workerSafetyModelSamples = std::max(calibratedHops * kModelBlockSize,
                                        blockEquivalentModelSamples + 4 * kModelBlockSize);
    workerSafetyModelSamples = juce::jlimit(baseReservoirHops * kModelBlockSize,
                                             maxReservoirHops * kModelBlockSize, workerSafetyModelSamples);
    safetyLatencyHostSamples = (int) std::ceil(hostSampleRate * ((double) workerSafetyModelSamples / kModelSampleRate));
    std::array<float, DiracGNNEngine::kNumOutputChannels> zeros {};
    for (int i = 0; i < safetyLatencyHostSamples; ++i)
        hostOutputFifo.pushFrame(zeros.data());

    setLatencySamples(calculateLatencySamples(hostSampleRate));

    modelWorker = std::make_unique<ModelWorker>(*this);
    modelWorker->startThread(juce::Thread::Priority::high);
}

void GNNReconstructorAudioProcessor::releaseResources()
{
    if (modelWorker != nullptr)
    {
        modelWorker->signalThreadShouldExit();
        workerWakeEvent.signal();
        modelWorker->stopThread(2000);
        modelWorker.reset();
    }
    gnnEngine.reset();
}

#ifndef JucePlugin_PreferredChannelConfigurations
bool GNNReconstructorAudioProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
    // Stable manual-I/O contract: one 40-channel discrete input bus and one
    // 4-channel FOA output bus. N is a logical DSP parameter, not a bus-layout
    // negotiation. This is intentionally the same flat routing family that was
    // stable across supported host configurations.
    return layouts.inputBuses.size() == 1
        && layouts.outputBuses.size() == 1
        && layouts.getMainInputChannelSet().size() == gnn4foa::ManualFoaPolicy::kPhysicalInputChannels
        && layouts.getMainOutputChannelSet().size() == gnn4foa::ManualFoaPolicy::kOutputChannels;
}
#endif

int GNNReconstructorAudioProcessor::calculateLatencySamples(double sampleRate) const noexcept
{
    const double seconds = (double) gnnEngine.getLatencySamples() / kModelSampleRate
                         + inputResampler.getDelaySeconds()
                         + outputResampler.getDelaySeconds()
                         + (double) workerSafetyModelSamples / kModelSampleRate;
    return (int) std::ceil(seconds * sampleRate);
}

void GNNReconstructorAudioProcessor::resetDiagnostics() noexcept
{
    modelBlocksProcessed.store(0, std::memory_order_relaxed);
    deadlineMisses.store(0, std::memory_order_relaxed);
    inputFifoOverflows.store(0, std::memory_order_relaxed);
    outputFifoOverflows.store(0, std::memory_order_relaxed);
    outputStarvations.store(0, std::memory_order_relaxed);
    lateOutputFramesDropped.store(0, std::memory_order_relaxed);
    outputBacklogFramesTrimmed.store(0, std::memory_order_relaxed);
    outputTimelineDropDebt.store(0, std::memory_order_relaxed);
    minOutputFifoDepth.store(0x7fffffff, std::memory_order_relaxed);
    lastWorkerMs.store(0.0, std::memory_order_relaxed);
    maxWorkerMs.store(0.0, std::memory_order_relaxed);
    outputReservoirFill.store(1.0f, std::memory_order_relaxed);
    activeWorkerMode.store(-1, std::memory_order_relaxed);
}

GNNReconstructorAudioProcessor::AudioSpatialSnapshot
GNNReconstructorAudioProcessor::readSpatialSnapshot() const noexcept
{
    AudioSpatialSnapshot out;
    for (int tries = 0; tries < 4; ++tries)
    {
        const uint32_t before = spatialStateVersion.load(std::memory_order_acquire);
        if ((before & 1u) != 0u)
            continue;

        for (int i = 0; i < kNumMics; ++i)
        {
            out.micX[(size_t) i] = micXAtomic[(size_t) i].load(std::memory_order_relaxed);
            out.micY[(size_t) i] = micYAtomic[(size_t) i].load(std::memory_order_relaxed);
            out.micZ[(size_t) i] = micZAtomic[(size_t) i].load(std::memory_order_relaxed);
            out.micAzimuth[(size_t) i] = micAzimuthAtomic[(size_t) i].load(std::memory_order_relaxed);
        }
        out.listenerX = listenerXAtomic.load(std::memory_order_relaxed);
        out.listenerY = listenerYAtomic.load(std::memory_order_relaxed);
        out.listenerZ = listenerZAtomic.load(std::memory_order_relaxed);

        const uint32_t after = spatialStateVersion.load(std::memory_order_acquire);
        if (before == after && (after & 1u) == 0u)
            return out;
    }
    return out;
}

void GNNReconstructorAudioProcessor::processOneModelBlockOnWorker()
{
    workerOutputBuffer.clear();

    const auto spatial = readSpatialSnapshot();
    const float smooth = diracSmoothingParam->load(std::memory_order_relaxed);
    const float gateDb = energyThresholdParam->load(std::memory_order_relaxed);
    const int mode = juce::roundToInt(interpModeParam->load(std::memory_order_relaxed));
    const int temporal = juce::roundToInt(temporalWindowParam->load(std::memory_order_relaxed));
    activeWorkerMode.store(mode, std::memory_order_relaxed);

    const int outputDepth = hostOutputFifo.getAvailableFrames();
    const float reservoirFill = juce::jlimit(0.0f, 1.5f,
        (float) outputDepth / (float) std::max(1, safetyLatencyHostSamples));
    outputReservoirFill.store(reservoirFill, std::memory_order_relaxed);
    const float reservoirAvailableMs = (float) (1000.0 * (double) outputDepth
                                              / std::max(1.0, hostSampleRate));

    const double t0 = juce::Time::getMillisecondCounterHiRes();
    gnnEngine.process(workerInputBuffer,
                      workerOutputBuffer,
                      spatial.listenerX,
                      spatial.listenerY,
                      spatial.listenerZ,
                      spatial.micX,
                      spatial.micY,
                      spatial.micZ,
                      spatial.micAzimuth,
                      smooth,
                      gateDb,
                      mode,
                      temporal,
                      effectiveMicCount.load(std::memory_order_relaxed),
                      reservoirFill,
                      reservoirAvailableMs);

    // Output SRC also runs on this worker. The audio callback only copies from
    // the final host-rate FIFO, so its workload is bounded and independent of
    // how much model output accumulated between host callbacks.
    for (int i = 0; i < kModelBlockSize; ++i)
    {
        for (int ch = 0; ch < DiracGNNEngine::kNumOutputChannels; ++ch)
            modelOutputFrame[(size_t) ch] = workerOutputBuffer.getSample(ch, i);

        const int generated = outputResampler.processFrame(modelOutputFrame.data(),
                                                            outputResampleScratch.data(),
                                                            StreamingSincResampler::kMaxGeneratedFrames);
        for (int g = 0; g < generated; ++g)
        {
            // Never repay an underrun by
            // deleting the first recovered worker samples.  Doing so created
            // the audible freeze -> jump -> new-fluid-sequence behaviour on
            // slow machines.  Backlog is bounded separately and only trimmed
            // for genuinely pathological accumulation.
            const float* frame = outputResampleScratch.data()
                               + (size_t) g * DiracGNNEngine::kNumOutputChannels;
            if (!hostOutputFifo.pushFrame(frame))
                outputFifoOverflows.fetch_add(1, std::memory_order_relaxed);
        }
    }

    const double elapsed = juce::Time::getMillisecondCounterHiRes() - t0;
    lastWorkerMs.store(elapsed, std::memory_order_relaxed);
    double previousMax = maxWorkerMs.load(std::memory_order_relaxed);
    while (elapsed > previousMax
           && !maxWorkerMs.compare_exchange_weak(previousMax, elapsed, std::memory_order_relaxed)) {}
    if (elapsed > kModelDeadlineMs)
        deadlineMisses.fetch_add(1, std::memory_order_relaxed);
    modelBlocksProcessed.fetch_add(1, std::memory_order_relaxed);
}

void GNNReconstructorAudioProcessor::runModelWorker()
{
    // MXCSR/FP state is per-thread on Intel. Keep FTZ/DAZ active on the actual
    // GNN worker; the audio callback's ScopedNoDenormals does not propagate.
    juce::ScopedNoDenormals noDenormals;

    while (modelWorker != nullptr && !modelWorker->threadShouldExit())
    {
        bool didWork = false;
        int hostFramesProcessed = 0;

        // Bound one service pass so GUI/OS scheduling still gets a chance even
        // after an offline seek or a temporarily delayed worker wake-up.
        constexpr int kMaxHostFramesPerPass = 4096;
        while (hostFramesProcessed < kMaxHostFramesPerPass
               && hostInputFifo.popFrame(hostInputFrame.data()))
        {
            didWork = true;
            ++hostFramesProcessed;
            const int generated = inputResampler.processFrame(hostInputFrame.data(),
                                                               inputResampleScratch.data(),
                                                               StreamingSincResampler::kMaxGeneratedFrames);
            for (int g = 0; g < generated; ++g)
            {
                const float* modelFrame = inputResampleScratch.data()
                                        + (size_t) g * (size_t) configuredInputChannels;
                for (int ch = 0; ch < configuredInputChannels; ++ch)
                    workerInputBuffer.setSample(ch, workerModelInputFill, modelFrame[ch]);

                ++workerModelInputFill;
                if (workerModelInputFill == kModelBlockSize)
                {
                    processOneModelBlockOnWorker();
                    workerModelInputFill = 0;
                }
            }
        }

        if (!didWork)
            workerWakeEvent.wait(2);
        else if (hostInputFifo.getAvailableFrames() > 0)
            juce::Thread::yield();
    }
}

void GNNReconstructorAudioProcessor::pushHostInputsToWorkerFifo(const juce::AudioBuffer<float>& buffer) noexcept
{
    const int n = buffer.getNumSamples();
    bool pushedAny = false;

    // Flat, deterministic 1:1 routing: host input ch 0..39 map to internal ch
    // 0..39. The selected N only determines how many leading FOA quartets enter
    // the graph; it never changes or renegotiates the host bus.
    const int readableChannels = juce::jmin(DiracGNNEngine::kNumInputChannels, buffer.getNumChannels());
    deliveredInputChannels.store(readableChannels, std::memory_order_relaxed);

    const int active = getSelectedMicCount();
    effectiveMicCount.store(active, std::memory_order_relaxed);
    const int activeChannels = gnn4foa::ManualFoaPolicy::channelsForNodes(active);

    for (int i = 0; i < n; ++i)
    {
        for (int ch = 0; ch < DiracGNNEngine::kNumInputChannels; ++ch)
            hostInputFrame[(size_t) ch] = (ch < readableChannels && ch < activeChannels)
                                               ? buffer.getSample(ch, i)
                                               : 0.0f;

        if (!hostInputFifo.pushFrame(hostInputFrame.data()))
            inputFifoOverflows.fetch_add(1, std::memory_order_relaxed);
        else
            pushedAny = true;
    }

    if (pushedAny)
        workerWakeEvent.signal();
}

void GNNReconstructorAudioProcessor::trimExcessOutputBacklog() noexcept
{
    // Normal steady-state depth is the explicit scheduling prebuffer plus at
    // most a few host callbacks of burstiness. If an underrun or
    // scheduler stall ever leaves seconds of stale audio queued, discard the
    // oldest frames here and return to the fixed-latency timeline.  Every
    // discarded frame is counted and exposed in the GUI; this is a diagnostic
    // resynchronisation, never a hidden interpolation fallback.
    const int depth = hostOutputFifo.getAvailableFrames();
    const int burstAllowance = std::max(8 * hostMaxBlockSize,
                                        (int) std::ceil(1.000 * hostSampleRate));
    const int hardLimit = safetyLatencyHostSamples + burstAllowance;
    const int trimTarget = safetyLatencyHostSamples
                         + std::max(2 * hostMaxBlockSize,
                                    (int) std::ceil(0.500 * hostSampleRate));
    if (depth <= hardLimit)
        return;

    const int toDrop = std::max(0, depth - trimTarget);
    // O(1): never spend a callback looping over seconds of stale audio.
    const int dropped = hostOutputFifo.discardOldestFrames(toDrop);

    if (dropped > 0)
    {
        outputBacklogFramesTrimmed.fetch_add((uint64_t) dropped, std::memory_order_relaxed);
        // Fade the newly re-aligned stream in instead of hard-jumping from a
        // stale queued sample to the current timeline.
        starvationGain = 0.0f;
    }
}

void GNNReconstructorAudioProcessor::renderHostOutput(juce::AudioBuffer<float>& buffer) noexcept
{
    const int n = buffer.getNumSamples();
    const float rampStep = 1.0f / (float) juce::jmax(64,
        (int) std::round(0.010 * hostSampleRate)); // ~10-ms recovery envelope

    trimExcessOutputBacklog();

    const int depthAtCallbackStart = hostOutputFifo.getAvailableFrames();
    int observedMin = minOutputFifoDepth.load(std::memory_order_relaxed);
    while (depthAtCallbackStart < observedMin
           && !minOutputFifoDepth.compare_exchange_weak(observedMin, depthAtCallbackStart,
                                                        std::memory_order_relaxed)) {}

    for (int i = 0; i < n; ++i)
    {
        const bool haveFrame = hostOutputFifo.popFrame(hostOutputFrame.data());
        if (haveFrame)
        {
            starvationGain = std::min(1.0f, starvationGain + rampStep);
            for (int ch = 0; ch < DiracGNNEngine::kNumOutputChannels; ++ch)
            {
                const float y = hostOutputFrame[(size_t) ch] * starvationGain;
                buffer.setSample(ch, i, y);
                lastRenderedSample[(size_t) ch] = y;
            }
        }
        else
        {
            outputStarvations.fetch_add(1, std::memory_order_relaxed);
            minOutputFifoDepth.store(0, std::memory_order_relaxed);
            // Timeline lag is exposed as a diagnostic only. The worker
            // no longer consumes this value by discarding recovered audio.
            outputTimelineDropDebt.fetch_add(1, std::memory_order_relaxed);
            starvationGain = std::max(0.0f, starvationGain - rampStep);
            for (int ch = 0; ch < DiracGNNEngine::kNumOutputChannels; ++ch)
            {
                const float y = lastRenderedSample[(size_t) ch] * starvationGain;
                buffer.setSample(ch, i, y);
            }
        }
    }
}

void GNNReconstructorAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer,
                                                   juce::MidiBuffer&)
{
    juce::ScopedNoDenormals noDenormals;

    if (getTotalNumOutputChannels() != 4)
    {
        for (int ch = 0; ch < juce::jmin(4, buffer.getNumChannels()); ++ch)
            buffer.clear(ch, 0, buffer.getNumSamples());
        return;
    }

    // All 40 input pins belong to one stable discrete bus. Read all selected
    // inputs before renderHostOutput() overwrites channels 0..3 with the result.
    pushHostInputsToWorkerFifo(buffer);
    renderHostOutput(buffer);
}

bool GNNReconstructorAudioProcessor::hasEditor() const { return true; }
juce::AudioProcessorEditor* GNNReconstructorAudioProcessor::createEditor()
{
    return new GNNReconstructorAudioProcessorEditor(*this);
}

int GNNReconstructorAudioProcessor::graphIndexForNodeId(int nodeId) const noexcept
{
    if (nodeId == listener.id) return kNumMics;
    for (int i = 0; i < (int) micArrays.size(); ++i)
        if (micArrays[(size_t) i].id == nodeId) return i;
    return -1;
}

void GNNReconstructorAudioProcessor::publishSpatialStateFromGui() noexcept
{
    if ((int) micArrays.size() < kNumMics)
        return;

    spatialStateVersion.fetch_add(1u, std::memory_order_acq_rel);
    for (int i = 0; i < kNumMics; ++i)
    {
        micXAtomic[(size_t) i].store(micArrays[(size_t) i].x, std::memory_order_relaxed);
        micYAtomic[(size_t) i].store(micArrays[(size_t) i].y, std::memory_order_relaxed);
        micZAtomic[(size_t) i].store(micArrays[(size_t) i].z, std::memory_order_relaxed);
        micAzimuthAtomic[(size_t) i].store(micArrays[(size_t) i].azimuth, std::memory_order_relaxed);
    }
    listenerXAtomic.store(listener.x, std::memory_order_relaxed);
    listenerYAtomic.store(listener.y, std::memory_order_relaxed);
    listenerZAtomic.store(listener.z, std::memory_order_relaxed);
    spatialStateVersion.fetch_add(1u, std::memory_order_release);
}

GNNReconstructorAudioProcessor::DiagnosticsSnapshot
GNNReconstructorAudioProcessor::getDiagnosticsSnapshot() const noexcept
{
    DiagnosticsSnapshot d;
    d.hostSampleRate = hostSampleRate;
    d.lastWorkerMs = lastWorkerMs.load(std::memory_order_relaxed);
    d.maxWorkerMs = maxWorkerMs.load(std::memory_order_relaxed);
    d.modelBlocks = modelBlocksProcessed.load(std::memory_order_relaxed);
    d.deadlineMisses = deadlineMisses.load(std::memory_order_relaxed);
    d.inputFifoOverflows = inputFifoOverflows.load(std::memory_order_relaxed);
    d.outputFifoOverflows = outputFifoOverflows.load(std::memory_order_relaxed);
    d.outputStarvations = outputStarvations.load(std::memory_order_relaxed);
    d.lateOutputFramesDropped = lateOutputFramesDropped.load(std::memory_order_relaxed);
    d.outputBacklogFramesTrimmed = outputBacklogFramesTrimmed.load(std::memory_order_relaxed);
    d.outputTimelineDropDebt = outputTimelineDropDebt.load(std::memory_order_relaxed);
    d.invalidGNNFrames = gnnEngine.getInvalidGNNFrameCount();
    d.reusedGNNFrames = gnnEngine.getReusedGNNFrameCount();
    d.lastGNNStage = gnnEngine.getLastGNNStage();
    d.lastMaxGNNWeight = gnnEngine.getLastMaxGNNWeight();
    const auto weightDiag = gnnEngine.getWeightVisualizationSnapshot();
    d.meanRawGNNWeightSum = weightDiag.meanRawSum;
    d.meanAppliedGNNWeightSum = weightDiag.meanAppliedSum;
    d.meanGNNWeightScale = weightDiag.meanScale;
    d.maxGNNWeightScale = weightDiag.maxScale;
    d.lowClampFraction = weightDiag.lowClampFraction;
    d.highClampFraction = weightDiag.highClampFraction;
    d.gnnBinFraction = weightDiag.gnnBinFraction;
    d.resamplingActive = resamplingActive;
    d.activeWorkerMode = activeWorkerMode.load(std::memory_order_relaxed);
    d.hostInputFifoDepth = hostInputFifo.getAvailableFrames();
    d.hostOutputFifoDepth = hostOutputFifo.getAvailableFrames();
    d.workerPrebufferMs = 1000.0 * (double) workerSafetyModelSamples / kModelSampleRate;
    d.outputQueueMs = 1000.0 * (double) d.hostOutputFifoDepth / std::max(1.0, hostSampleRate);
    const int minDepthRaw = minOutputFifoDepth.load(std::memory_order_relaxed);
    const int minDepth = minDepthRaw == 0x7fffffff ? d.hostOutputFifoDepth : minDepthRaw;
    d.minOutputQueueMs = 1000.0 * (double) std::max(0, minDepth) / std::max(1.0, hostSampleRate);
    const int nominalQueue = safetyLatencyHostSamples + hostMaxBlockSize;
    d.excessOutputQueueMs = 1000.0 * (double) std::max(0, d.hostOutputFifoDepth - nominalQueue)
                          / std::max(1.0, hostSampleRate);
    const double nominalPreloadMs = 1000.0 * (double) safetyLatencyHostSamples
                                  / std::max(1.0, hostSampleRate);
    d.timelineCurrentlyHealthy = d.outputQueueMs > 0.10 * nominalPreloadMs
                              && d.excessOutputQueueMs < 1000.0;
    d.inputChannels = configuredInputChannels;
    d.deliveredInputChannels = deliveredInputChannels.load(std::memory_order_relaxed);
    d.configuredMicCount = configuredMicCount;
    d.activeMicCount = effectiveMicCount.load(std::memory_order_relaxed);
    d.nodeCountMode = getSelectedMicCount();
    d.lastGNNForwardMs = gnnEngine.getLastGNNForwardMs();
    d.gnnForwardEwmaMs = gnnEngine.getGNNForwardEwmaMs();
    d.gnnInferenceStride = gnnEngine.getGNNInferenceStride();
    d.gnnActiveBinCount = gnnEngine.getLastGNNActiveBinCount();
    d.gnnInferenceCount = gnnEngine.getGNNInferenceCount();
    d.gnnCadenceHeldFrames = gnnEngine.getGNNCadenceHeldFrameCount();
    d.gnnPressureDeferredFrames = gnnEngine.getGNNPressureDeferredFrameCount();
    d.gnnGeometryForcedInferences = gnnEngine.getGNNGeometryForcedInferenceCount();
    d.gnnAcousticForcedInferences = gnnEngine.getGNNAcousticForcedInferenceCount();
    d.gnnGateEntries = gnnEngine.getGNNGateEnterCount();
    d.gnnGateExits = gnnEngine.getGNNGateExitCount();
    d.gnnCoverageOnlyInferences = gnnEngine.getGNNCoverageInferenceCount();
    d.gnnFullInferences = gnnEngine.getGNNFullInferenceCount();
    d.gnnLastInferredBinCount = gnnEngine.getLastGNNInferredBinCount();
    d.gnnCoverageBinCount = gnnEngine.getLastGNNCoverageBinCount();
    d.gnnPredictedForwardMs = gnnEngine.getLastPredictedGNNForwardMs();
    d.gnnForwardPerBinEwmaMs = gnnEngine.getGNNForwardPerBinEwmaMs();
    d.gnnAcousticNovelty = gnnEngine.getLastAcousticNovelty();
    d.outputReservoirFill = outputReservoirFill.load(std::memory_order_relaxed);
    d.startupGNNBenchmarkMs = startupGNNBenchmarkMs;
    d.adaptivePrebufferMs = 1000.0 * (double) workerSafetyModelSamples / kModelSampleRate;
    return d;
}

void GNNReconstructorAudioProcessor::appendSpatialStateToValueTree(juce::ValueTree& state) const
{
    state.setProperty(kStateVersionProperty, kSpatialStateVersion, nullptr);
    for (int i = 0; i < kNumMics && i < (int) micArrays.size(); ++i)
    {
        const auto prefix = "MIC_" + juce::String(i) + "_";
        state.setProperty(prefix + "X", micArrays[(size_t) i].x, nullptr);
        state.setProperty(prefix + "Y", micArrays[(size_t) i].y, nullptr);
        state.setProperty(prefix + "Z", micArrays[(size_t) i].z, nullptr);
        state.setProperty(prefix + "AZ", micArrays[(size_t) i].azimuth, nullptr);
    }
    state.setProperty("LISTENER_X", listener.x, nullptr);
    state.setProperty("LISTENER_Y", listener.y, nullptr);
    state.setProperty("LISTENER_Z", listener.z, nullptr);
    // The reconstructed/query node intentionally has no orientation. Older
    // sessions may still contain LISTENER_AZ; it is ignored on restore.

    state.setProperty("MAP_LOCKED", mapSceneLocked, nullptr);
    state.setProperty("MAP_VIEW_CX", mapViewCenterX, nullptr);
    state.setProperty("MAP_VIEW_CY", mapViewCenterY, nullptr);
    state.setProperty("MAP_VIEW_ZOOM", mapViewZoom, nullptr);

    juce::StringArray annotationTokens;
    for (const auto& annotation : mapAnnotations)
    {
        juce::String token;
        switch (annotation.type)
        {
            case MapAnnotation::Type::wallSegment: token = "W"; break;
            case MapAnnotation::Type::source:       token = "S"; break;
            case MapAnnotation::Type::dashedStroke: token = "D"; break;
        }
        for (const auto& pt : annotation.points)
            token << "," << juce::String(pt.x, 6) << "," << juce::String(pt.y, 6);
        annotationTokens.add(token);
    }
    state.setProperty("MAP_ANNOTATIONS", annotationTokens.joinIntoString(";"), nullptr);

    juce::StringArray edgeTokens;
    for (const auto& edge : customEdges)
        edgeTokens.add(juce::String(edge.first) + "," + juce::String(edge.second));
    state.setProperty("CUSTOM_EDGES", edgeTokens.joinIntoString(";"), nullptr);
}

void GNNReconstructorAudioProcessor::restoreSpatialStateFromValueTree(const juce::ValueTree& state)
{
    if ((int) micArrays.size() < kNumMics)
        return;

    for (int i = 0; i < kNumMics; ++i)
    {
        auto& mic = micArrays[(size_t) i];
        const auto prefix = "MIC_" + juce::String(i) + "_";
        mic.x = (float) state.getProperty(prefix + "X", mic.x);
        mic.y = (float) state.getProperty(prefix + "Y", mic.y);
        mic.z = (float) state.getProperty(prefix + "Z", mic.z);
        mic.azimuth = (float) state.getProperty(prefix + "AZ", mic.azimuth);
    }
    listener.x = (float) state.getProperty("LISTENER_X", listener.x);
    listener.y = (float) state.getProperty("LISTENER_Y", listener.y);
    listener.z = (float) state.getProperty("LISTENER_Z", listener.z);
    listener.azimuth = 0.0f; // query orientation is deliberately undefined/unused

    mapSceneLocked = (bool) state.getProperty("MAP_LOCKED", false);
    mapViewCenterX = (float) state.getProperty("MAP_VIEW_CX", 0.0f);
    mapViewCenterY = (float) state.getProperty("MAP_VIEW_CY", 0.0f);
    mapViewZoom = juce::jlimit(0.01f, 20.0f, (float) state.getProperty("MAP_VIEW_ZOOM", 1.0f));
    mapAnnotations.clear();
    juce::StringArray annotationTokens;
    annotationTokens.addTokens(state.getProperty("MAP_ANNOTATIONS", juce::String()).toString(), ";", "");
    for (const auto& token : annotationTokens)
    {
        juce::StringArray fields;
        fields.addTokens(token, ",", "");
        if (fields.size() < 3)
            continue;

        MapAnnotation annotation;
        if (fields[0] == "W") annotation.type = MapAnnotation::Type::wallSegment;
        else if (fields[0] == "S") annotation.type = MapAnnotation::Type::source;
        else if (fields[0] == "D") annotation.type = MapAnnotation::Type::dashedStroke;
        else continue;

        for (int i = 1; i + 1 < fields.size(); i += 2)
            annotation.points.emplace_back(fields[i].getFloatValue(), fields[i + 1].getFloatValue());

        const bool valid = (annotation.type == MapAnnotation::Type::source && annotation.points.size() >= 1)
                        || (annotation.type != MapAnnotation::Type::source && annotation.points.size() >= 2);
        if (valid)
            mapAnnotations.push_back(std::move(annotation));
    }

    customEdges.clear();
    const auto edgeString = state.getProperty("CUSTOM_EDGES", juce::String()).toString();
    juce::StringArray edges;
    edges.addTokens(edgeString, ";", "");
    for (const auto& token : edges)
    {
        juce::StringArray ids;
        ids.addTokens(token, ",", "");
        if (ids.size() != 2) continue;
        int a = ids[0].getIntValue();
        int b = ids[1].getIntValue();
        if (graphIndexForNodeId(a) < 0 || graphIndexForNodeId(b) < 0 || a == b) continue;
        if (a > b) std::swap(a, b);
        const auto edge = std::make_pair(a, b);
        if (std::find(customEdges.begin(), customEdges.end(), edge) == customEdges.end())
            customEdges.push_back(edge);
    }
    publishSpatialStateFromGui();
}

void GNNReconstructorAudioProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    auto state = apvts.copyState();
    appendSpatialStateToValueTree(state);
    if (auto xml = state.createXml())
        copyXmlToBinary(*xml, destData);
}

void GNNReconstructorAudioProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    auto xmlState = getXmlFromBinary(data, sizeInBytes);
    if (xmlState == nullptr) return;
    auto state = juce::ValueTree::fromXml(*xmlState);
    if (!state.isValid() || !state.hasType(apvts.state.getType())) return;
    const int savedSpatialVersion = (int) state.getProperty(kStateVersionProperty, 0);
    if (savedSpatialVersion < 9)
        migrateNodeCountParameter(state, savedSpatialVersion);
    apvts.replaceState(state);
    restoreSpatialStateFromValueTree(state);
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new GNNReconstructorAudioProcessor();
}
