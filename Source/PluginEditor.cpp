// GNN4FOA v2
// Authors: Gioele Greco, Filippo Marri, Mirco Pezzoli, Fabio Antonacci
// Image and Sound Processing Lab (ISPL), Politecnico di Milano
// Copyright (c) 2026 Gioele Greco, Filippo Marri, Mirco Pezzoli, Fabio Antonacci

#include "PluginProcessor.h"
#include "PluginEditor.h"

#include <cmath>

namespace
{
constexpr float kPi = juce::MathConstants<float>::pi;
constexpr float radiansToDegrees(float radians) noexcept { return radians * 180.0f / kPi; }
constexpr float degreesToRadians(float degrees) noexcept { return degrees * kPi / 180.0f; }

float chooseGridStep(float pixelsPerMeter)
{
    const float raw = 48.0f / juce::jmax(0.0001f, pixelsPerMeter);
    const float exponent = std::floor(std::log10(juce::jmax(raw, 1.0e-8f)));
    const float decade = std::pow(10.0f, exponent);
    const float normalized = raw / decade;
    const float nice = normalized <= 1.0f ? 1.0f : (normalized <= 2.0f ? 2.0f : (normalized <= 5.0f ? 5.0f : 10.0f));
    return nice * decade;
}

juce::Path makeStar(juce::Point<float> centre, float outerRadius, float innerRadius)
{
    juce::Path star;
    for (int i = 0; i < 10; ++i)
    {
        const float angle = -0.5f * kPi + (float) i * kPi / 5.0f;
        const float radius = (i & 1) == 0 ? outerRadius : innerRadius;
        const juce::Point<float> p(centre.x + radius * std::cos(angle),
                                   centre.y + radius * std::sin(angle));
        if (i == 0) star.startNewSubPath(p); else star.lineTo(p);
    }
    star.closeSubPath();
    return star;
}
}

GNNReconstructorAudioProcessorEditor::GNNReconstructorAudioProcessorEditor(
    GNNReconstructorAudioProcessor& p)
    : AudioProcessorEditor(&p), audioProcessor(p)
{
    setSize(1000, 760);
    setWantsKeyboardFocus(true);

    viewCenterX = audioProcessor.mapViewCenterX;
    viewCenterY = audioProcessor.mapViewCenterY;
    zoomFactor = juce::jlimit(0.01f, 20.0f, audioProcessor.mapViewZoom);

    interpSelector.addItem("Delaunay Baseline", 1);
    interpSelector.addItem("GNN Learned", 2);
    addAndMakeVisible(interpSelector);

    nodeCountSelector.addItem("1 FOA node = inputs 1-4", 1);
    nodeCountSelector.addItem("2 FOA nodes = inputs 1-8", 2);
    nodeCountSelector.addItem("3 FOA nodes = inputs 1-12", 3);
    nodeCountSelector.addItem("4 FOA nodes = inputs 1-16", 4);
    nodeCountSelector.addItem("5 FOA nodes = inputs 1-20", 5);
    nodeCountSelector.addItem("6 FOA nodes = inputs 1-24", 6);
    nodeCountSelector.addItem("7 FOA nodes = inputs 1-28", 7);
    nodeCountSelector.addItem("8 FOA nodes = inputs 1-32", 8);
    nodeCountSelector.addItem("9 FOA nodes = inputs 1-36", 9);
    nodeCountSelector.addItem("10 FOA nodes = inputs 1-40", 10);
    addAndMakeVisible(nodeCountSelector);

    auto setupSlider = [this](juce::Slider& slider)
    {
        slider.setSliderStyle(juce::Slider::LinearHorizontal);
        slider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 72, 22);
        addAndMakeVisible(slider);
    };

    setupSlider(thresholdSlider);
    setupSlider(smoothingSlider);
    setupSlider(temporalWindowSlider);
    temporalWindowSlider.setNumDecimalPlacesToDisplay(0);

    interpAttach = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(
        audioProcessor.apvts, "INTERP_MODE", interpSelector);
    nodeCountAttach = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(
        audioProcessor.apvts, "NODE_COUNT", nodeCountSelector);
    threshAttach = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.apvts, "ENG_THRESH", thresholdSlider);
    smoothAttach = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.apvts, "DIRAC_SMOOTH", smoothingSlider);
    temporalWindowAttach = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.apvts, "GNN_WINDOW", temporalWindowSlider);

    auto setupMapButton = [this](juce::TextButton& button)
    {
        addAndMakeVisible(button);
        button.setColour(juce::TextButton::buttonColourId, juce::Colour(0xff30343a));
        button.setColour(juce::TextButton::buttonOnColourId, juce::Colour(0xff557088));
        button.setColour(juce::TextButton::textColourOffId, juce::Colours::white);
        button.setColour(juce::TextButton::textColourOnId, juce::Colours::white);
    };

    setupMapButton(fitViewButton);
    setupMapButton(panToolButton);
    setupMapButton(moveToolButton);
    setupMapButton(wallToolButton);
    setupMapButton(sourceToolButton);
    setupMapButton(dashedToolButton);
    setupMapButton(undoAnnotationButton);
    setupMapButton(lockSceneButton);

    for (auto* button : { &panToolButton, &moveToolButton, &wallToolButton, &sourceToolButton, &dashedToolButton })
    {
        button->setClickingTogglesState(true);
        button->setRadioGroupId(42001);
    }
    panToolButton.setToggleState(true, juce::dontSendNotification);
    lockSceneButton.setClickingTogglesState(true);
    lockSceneButton.setToggleState(audioProcessor.mapSceneLocked, juce::dontSendNotification);

    fitViewButton.onClick = [this] { fitMapView(); };
    panToolButton.onClick = [this] { setMapTool(MapTool::pan); };
    moveToolButton.onClick = [this] { setMapTool(MapTool::moveNode); };
    wallToolButton.onClick = [this] { setMapTool(MapTool::wall); };
    sourceToolButton.onClick = [this] { setMapTool(MapTool::source); };
    dashedToolButton.onClick = [this] { setMapTool(MapTool::dashed); };
    undoAnnotationButton.onClick = [this]
    {
        if (!audioProcessor.mapSceneLocked && !audioProcessor.mapAnnotations.empty())
        {
            audioProcessor.mapAnnotations.pop_back();
            updateToolState();
            repaint(mapArea);
        }
    };
    lockSceneButton.onClick = [this]
    {
        audioProcessor.mapSceneLocked = lockSceneButton.getToggleState();
        if (audioProcessor.mapSceneLocked)
            selectedNode = &audioProcessor.listener;
        drawingAnnotation = false;
        updateToolState();
        updatePropertiesPanel();
        repaint();
    };

    addAndMakeVisible(propTitle);
    propTitle.setText("Node Settings (click a node)", juce::dontSendNotification);
    propTitle.setJustificationType(juce::Justification::centredLeft);
    propTitle.setColour(juce::Label::textColourId, juce::Colours::white);

    auto setupLabel = [this](juce::Label& lbl,
                             juce::Label& edit,
                             const juce::String& text)
    {
        addAndMakeVisible(lbl);
        lbl.setText(text, juce::dontSendNotification);
        lbl.setColour(juce::Label::textColourId, juce::Colours::white);

        addAndMakeVisible(edit);
        edit.setEditable(true);
        edit.setColour(juce::Label::backgroundColourId, juce::Colours::black);
        edit.setColour(juce::Label::textColourId, juce::Colours::white);
        edit.setColour(juce::Label::outlineColourId, juce::Colours::grey);

        edit.onTextChange = [this, &edit]()
        {
            if (selectedNode == nullptr)
                return;

            const bool listener = selectedNode->isListener;
            if (audioProcessor.mapSceneLocked && !listener)
                return;
            if (&edit == &azEdit && listener)
                return;

            const float value = edit.getText().getFloatValue();
            if (&edit == &xEdit)  selectedNode->x = value;
            if (&edit == &yEdit)  selectedNode->y = value;
            if (&edit == &zEdit)  selectedNode->z = value;
            if (&edit == &azEdit) selectedNode->azimuth = degreesToRadians(value);
            publishSpatialEdit();
            repaint();
        };
    };

    setupLabel(xLabel, xEdit, "X (m):");
    setupLabel(yLabel, yEdit, "Y (m):");
    setupLabel(zLabel, zEdit, "Z (m):");
    setupLabel(azLabel, azEdit, "Mic yaw (deg):");

    addAndMakeVisible(diagnosticsLabel);
    diagnosticsLabel.setColour(juce::Label::textColourId, juce::Colours::white);
    diagnosticsLabel.setColour(juce::Label::backgroundColourId, juce::Colours::black.withAlpha(0.45f));
    diagnosticsLabel.setJustificationType(juce::Justification::topLeft);
    diagnosticsLabel.setFont(11.5f);

    updateToolState();
    updatePropertiesPanel();
    audioProcessor.publishSpatialStateFromGui();
    // The worker publishes a compact lock-free weight snapshot; the editor reads
    // it at 4 Hz. Map annotations never enter the DSP.
    startTimerHz(4);
    lastDiagnosticsUpdateMs = juce::Time::getMillisecondCounterHiRes();
}

GNNReconstructorAudioProcessorEditor::~GNNReconstructorAudioProcessorEditor()
{
    stopTimer();
    storeMapViewState();
}

void GNNReconstructorAudioProcessorEditor::publishSpatialEdit() noexcept
{
    audioProcessor.publishSpatialStateFromGui();
}

bool GNNReconstructorAudioProcessorEditor::shouldHandleWalkingKeys() const noexcept
{
    // Map click explicitly gives the editor focus.  Do not steal arrows from
    // sliders, combo boxes or editable coordinate fields.
    return hasKeyboardFocus(false);
}

bool GNNReconstructorAudioProcessorEditor::anyWalkingArrowDown() const noexcept
{
    return juce::KeyPress::isKeyCurrentlyDown(juce::KeyPress::leftKey)
        || juce::KeyPress::isKeyCurrentlyDown(juce::KeyPress::rightKey)
        || juce::KeyPress::isKeyCurrentlyDown(juce::KeyPress::upKey)
        || juce::KeyPress::isKeyCurrentlyDown(juce::KeyPress::downKey);
}

void GNNReconstructorAudioProcessorEditor::setWalkingTimerHighRate(bool highRate)
{
    if (walkHighRateTimer == highRate)
        return;

    walkHighRateTimer = highRate;
    if (highRate)
    {
        lastWalkTickMs = juce::Time::getMillisecondCounterHiRes();
        startTimerHz(kWalkTimerHz);
    }
    else
        startTimerHz(4);
}

void GNNReconstructorAudioProcessorEditor::softlyKeepListenerVisible()
{
    if (mapArea.isEmpty())
        return;

    const float ppm = 20.0f * zoomFactor;
    if (ppm <= 0.0001f)
        return;

    const auto p = mapToPixel({ audioProcessor.listener.x, audioProcessor.listener.y });
    const auto safe = mapArea.toFloat().reduced(58.0f, 52.0f);
    const float wantedX = juce::jlimit(safe.getX(), safe.getRight(), p.x);
    const float wantedY = juce::jlimit(safe.getY(), safe.getBottom(), p.y);
    const float errorX = p.x - wantedX;
    const float errorY = p.y - wantedY;

    if (std::abs(errorX) > 0.01f || std::abs(errorY) > 0.01f)
    {
        constexpr float follow = 0.22f;
        viewCenterX += follow * errorX / ppm;
        viewCenterY -= follow * errorY / ppm;
        storeMapViewState();
    }
}

void GNNReconstructorAudioProcessorEditor::updateKeyboardWalking(double nowMs)
{
    if (lastWalkTickMs <= 0.0)
        lastWalkTickMs = nowMs;

    const float dt = (float) juce::jlimit(0.0, 0.050, (nowMs - lastWalkTickMs) * 0.001);
    lastWalkTickMs = nowMs;

    float ix = 0.0f, iy = 0.0f;
    if (shouldHandleWalkingKeys())
    {
        if (juce::KeyPress::isKeyCurrentlyDown(juce::KeyPress::leftKey))  ix -= 1.0f;
        if (juce::KeyPress::isKeyCurrentlyDown(juce::KeyPress::rightKey)) ix += 1.0f;
        if (juce::KeyPress::isKeyCurrentlyDown(juce::KeyPress::upKey))    iy += 1.0f;
        if (juce::KeyPress::isKeyCurrentlyDown(juce::KeyPress::downKey))  iy -= 1.0f;
    }

    const float norm = std::sqrt(ix * ix + iy * iy);
    if (norm > 1.0f)
    {
        ix /= norm;
        iy /= norm;
    }

    const bool precision = juce::ModifierKeys::getCurrentModifiersRealtime().isShiftDown();
    const float speed = precision ? kPrecisionWalkSpeedMetersPerSecond
                                  : kWalkSpeedMetersPerSecond;
    const float targetVX = ix * speed;
    const float targetVY = iy * speed;
    const bool hasInput = norm > 0.0f;
    const float tau = hasInput ? 0.115f : 0.075f;
    const float alpha = dt > 0.0f ? 1.0f - std::exp(-dt / tau) : 0.0f;

    walkVelocityX += alpha * (targetVX - walkVelocityX);
    walkVelocityY += alpha * (targetVY - walkVelocityY);
    if (!hasInput && std::hypot(walkVelocityX, walkVelocityY) < 0.004f)
        walkVelocityX = walkVelocityY = 0.0f;

    if (dt > 0.0f && (walkVelocityX != 0.0f || walkVelocityY != 0.0f))
    {
        audioProcessor.listener.x += walkVelocityX * dt;
        audioProcessor.listener.y += walkVelocityY * dt;
        selectedNode = &audioProcessor.listener;
        publishSpatialEdit();
        softlyKeepListenerVisible();

        if (++walkPropertyRefreshDivider >= 5) // 10 Hz is ample for text fields.
        {
            walkPropertyRefreshDivider = 0;
            updatePropertiesPanel();
        }
        repaint(mapArea);
    }

    if (!hasInput && walkVelocityX == 0.0f && walkVelocityY == 0.0f)
    {
        updatePropertiesPanel();
        setWalkingTimerHighRate(false);
    }
}

bool GNNReconstructorAudioProcessorEditor::keyPressed(const juce::KeyPress& key)
{
    if (!shouldHandleWalkingKeys())
        return false;

    const int code = key.getKeyCode();
    const bool arrow = code == juce::KeyPress::leftKey || code == juce::KeyPress::rightKey
                    || code == juce::KeyPress::upKey || code == juce::KeyPress::downKey;
    if (!arrow)
        return false;

    selectedNode = &audioProcessor.listener;
    setWalkingTimerHighRate(true);
    updatePropertiesPanel();
    repaint(mapArea);
    return true;
}

bool GNNReconstructorAudioProcessorEditor::keyStateChanged(bool isKeyDown)
{
    juce::ignoreUnused(isKeyDown);
    if (!shouldHandleWalkingKeys())
        return false;

    if (anyWalkingArrowDown() || walkVelocityX != 0.0f || walkVelocityY != 0.0f)
    {
        setWalkingTimerHighRate(true);
        return true;
    }
    return false;
}

void GNNReconstructorAudioProcessorEditor::storeMapViewState() noexcept
{
    audioProcessor.mapViewCenterX = viewCenterX;
    audioProcessor.mapViewCenterY = viewCenterY;
    audioProcessor.mapViewZoom = zoomFactor;
}

juce::Point<float> GNNReconstructorAudioProcessorEditor::mapToPixel(juce::Point<float> p) const noexcept
{
    const float ppm = 20.0f * zoomFactor;
    return { (float) mapArea.getCentreX() + (p.x - viewCenterX) * ppm,
             (float) mapArea.getCentreY() - (p.y - viewCenterY) * ppm };
}

juce::Point<float> GNNReconstructorAudioProcessorEditor::pixelToMap(juce::Point<float> p) const noexcept
{
    const float ppm = 20.0f * zoomFactor;
    return { viewCenterX + (p.x - (float) mapArea.getCentreX()) / ppm,
             viewCenterY - (p.y - (float) mapArea.getCentreY()) / ppm };
}

void GNNReconstructorAudioProcessorEditor::setMapTool(MapTool tool)
{
    // The hand is a viewport operation and is always allowed, even when the
    // scene is locked. Node mode remains available while locked, but hit-test
    // then exposes only the orange query node.
    if (audioProcessor.mapSceneLocked
        && tool != MapTool::pan
        && tool != MapTool::moveNode)
        tool = MapTool::pan;

    currentTool = tool;
    panToolButton.setToggleState(tool == MapTool::pan, juce::dontSendNotification);
    moveToolButton.setToggleState(tool == MapTool::moveNode, juce::dontSendNotification);
    wallToolButton.setToggleState(tool == MapTool::wall, juce::dontSendNotification);
    sourceToolButton.setToggleState(tool == MapTool::source, juce::dontSendNotification);
    dashedToolButton.setToggleState(tool == MapTool::dashed, juce::dontSendNotification);
    repaint(mapArea);
}

void GNNReconstructorAudioProcessorEditor::updateToolState()
{
    const bool locked = audioProcessor.mapSceneLocked;
    lockSceneButton.setToggleState(locked, juce::dontSendNotification);
    lockSceneButton.setButtonText(locked ? "Unlock Scene" : "Lock Scene");
    panToolButton.setEnabled(true);
    moveToolButton.setEnabled(true);
    wallToolButton.setEnabled(!locked);
    sourceToolButton.setEnabled(!locked);
    dashedToolButton.setEnabled(!locked);
    undoAnnotationButton.setEnabled(!locked && !audioProcessor.mapAnnotations.empty());
    if (locked && currentTool != MapTool::pan && currentTool != MapTool::moveNode)
        setMapTool(MapTool::pan);
}

void GNNReconstructorAudioProcessorEditor::commitAnnotation(MapAnnotation annotation)
{
    if (audioProcessor.mapSceneLocked)
        return;
    audioProcessor.mapAnnotations.push_back(std::move(annotation));
    updateToolState();
    repaint(mapArea);
}

void GNNReconstructorAudioProcessorEditor::fitMapView()
{
    bool havePoint = false;
    float minX = 0.0f, maxX = 0.0f, minY = 0.0f, maxY = 0.0f;
    auto include = [&](juce::Point<float> p)
    {
        if (!havePoint)
        {
            minX = maxX = p.x;
            minY = maxY = p.y;
            havePoint = true;
        }
        else
        {
            minX = juce::jmin(minX, p.x); maxX = juce::jmax(maxX, p.x);
            minY = juce::jmin(minY, p.y); maxY = juce::jmax(maxY, p.y);
        }
    };

    const int activeMics = juce::jmin(audioProcessor.getSceneMicCount(),
                                      (int) audioProcessor.micArrays.size());
    for (int m = 0; m < activeMics; ++m)
        include({ audioProcessor.micArrays[(size_t) m].x, audioProcessor.micArrays[(size_t) m].y });
    include({ audioProcessor.listener.x, audioProcessor.listener.y });
    for (const auto& annotation : audioProcessor.mapAnnotations)
        for (const auto& p : annotation.points) include(p);

    if (!havePoint || mapArea.isEmpty())
        return;

    viewCenterX = 0.5f * (minX + maxX);
    viewCenterY = 0.5f * (minY + maxY);
    const float spanX = juce::jmax(1.0f, maxX - minX);
    const float spanY = juce::jmax(1.0f, maxY - minY);
    const float ppmX = ((float) mapArea.getWidth() - 70.0f) / spanX;
    const float ppmY = ((float) mapArea.getHeight() - 70.0f) / spanY;
    zoomFactor = juce::jlimit(0.01f, 20.0f, 0.90f * juce::jmin(ppmX, ppmY) / 20.0f);
    storeMapViewState();
    repaint(mapArea);
}

void GNNReconstructorAudioProcessorEditor::timerCallback()
{
    const double nowMs = juce::Time::getMillisecondCounterHiRes();
    if (walkHighRateTimer || walkVelocityX != 0.0f || walkVelocityY != 0.0f)
        updateKeyboardWalking(nowMs);

    // Keep expensive diagnostic/text/history work at the original ~4 Hz even
    // while the locomotion integrator temporarily runs at 50 Hz.
    if (nowMs - lastDiagnosticsUpdateMs < 235.0)
        return;
    lastDiagnosticsUpdateMs = nowMs;

    const auto d = audioProcessor.getDiagnosticsSnapshot();
    const int activeMics = juce::jlimit(1, GNNReconstructorAudioProcessor::kNumMics,
                                        audioProcessor.getActiveMicCount());

    // The worker
    // publishes the compact statistics once per STFT frame and the message
    // thread samples them at 4 Hz. No GUI code is executed on the worker.
    const auto newSnapshot = audioProcessor.getWeightVisualizationSnapshot();
    bool haveNewWeightFrame = false;
    if (newSnapshot.frameCounter > 0
        && newSnapshot.frameCounter != lastWeightHistoryFrameCounter)
    {
        weightSnapshot = newSnapshot;
        haveNewWeightFrame = true;
        const int snapshotMode = weightSnapshot.gnnMode ? 1 : 0;
        if (lastWeightHistoryMode != snapshotMode)
        {
            weightHistoryWriteIndex = 0;
            weightHistoryCount = 0;
            lastWeightHistoryMode = snapshotMode;
        }

        for (int m = 0; m < GNNReconstructorAudioProcessor::kNumMics; ++m)
        {
            const float historyWeight = m < activeMics
                                      ? (weightSnapshot.gnnMode
                                           ? weightSnapshot.gnnRelativeMean[(size_t) m]
                                           : weightSnapshot.relativeMean[(size_t) m])
                                      : 0.0f;
            weightHistory[(size_t) m][(size_t) weightHistoryWriteIndex] =
                juce::jlimit(0.0f, 1.0f, historyWeight);
        }
        weightHistoryWriteIndex = (weightHistoryWriteIndex + 1) % kWeightHistoryLength;
        weightHistoryCount = juce::jmin(kWeightHistoryLength, weightHistoryCount + 1);
        lastWeightHistoryFrameCounter = weightSnapshot.frameCounter;
    }

    const bool historicalProblem = d.deadlineMisses > 0 || d.inputFifoOverflows > 0
                                || d.outputFifoOverflows > 0 || d.outputStarvations > 0
                                || d.invalidGNNFrames > 0 || d.outputBacklogFramesTrimmed > 0;
    const bool currentProblem = !d.timelineCurrentlyHealthy;
    diagnosticsLabel.setColour(juce::Label::textColourId,
                               currentProblem ? juce::Colours::orange : juce::Colours::lightgreen);

    juce::String text;
    const juce::String dspMode = d.activeWorkerMode == 1 ? "GNN"
                               : (d.activeWorkerMode == 0 ? "DELAUNAY" : "WAITING");
    text << audioProcessor.getBuildIdentity() << "\n"
         << (currentProblem ? "NOW: RECOVERING / CHECK TIMELINE" : "NOW: OK")
         << "   DSP: " << dspMode
         << (historicalProblem ? "   HISTORY: EVENTS LATCHED" : "") << "\n";
    text << "MANUAL I/O: " << d.activeMicCount << " FOA nodes = first "
         << (d.activeMicCount * 4) << " input channels"
         << "   PHYSICAL BUS=" << d.inputChannels << " in / 4 out"
         << "   [NO AUTO DETECTION / NO REAPER REMAP]\n";
    if (d.deliveredInputChannels < d.activeMicCount * 4)
        text << "INPUT WARNING: host buffer exposes only " << d.deliveredInputChannels
             << " channels; selected N needs " << (d.activeMicCount * 4) << "\n";
    text
         << "GNN GEMM: " << DiracGNNEngine::getMatrixBackendName()
         << "  stage " << d.lastGNNStage
         << "  forward actual/pred " << juce::String(d.lastGNNForwardMs, 2) << "/"
         << juce::String(d.gnnPredictedForwardMs, 2) << " ms"
         << "  bins " << d.gnnLastInferredBinCount
         << "  1/" << d.gnnInferenceStride << " ("
         << juce::String(125.0 / (double) juce::jmax(1, d.gnnInferenceStride), 1) << " Hz)\n"
         << "Startup GNN bench " << juce::String(d.startupGNNBenchmarkMs, 2) << " ms"
         << "  adaptive preload " << juce::String(d.adaptivePrebufferMs, 0) << " ms\n"
         << "HOST " << juce::String(d.hostSampleRate, 0) << " -> MODEL 16000 Hz"
         << (d.resamplingActive ? " [WORKER SRC]" : " [BYPASS]") << "\n"
         << "Worker " << juce::String(d.lastWorkerMs, 2) << " ms  max "
         << juce::String(d.maxWorkerMs, 2) << "  >8ms " << (juce::int64) d.deadlineMisses
         << "  GNN# " << (juce::int64) d.gnnInferenceCount << " held "
         << (juce::int64) d.gnnCadenceHeldFrames
         << "  acoustic-forced " << (juce::int64) d.gnnAcousticForcedInferences
         << " nov " << juce::String(d.gnnAcousticNovelty, 3) << "\n"
         << "Gate enter/exit " << (juce::int64) d.gnnGateEntries << "/" << (juce::int64) d.gnnGateExits
         << "  coverage/full " << (juce::int64) d.gnnCoverageOnlyInferences << "/"
         << (juce::int64) d.gnnFullInferences
         << "  missing " << d.gnnCoverageBinCount << " bins\n"
         << "Reservoir " << juce::String(100.0f * d.outputReservoirFill, 0) << "%  outQ "
         << d.hostOutputFifoDepth << " (" << juce::String(d.outputQueueMs, 1) << " ms)"
         << " min " << juce::String(d.minOutputQueueMs, 1) << " ms"
         << " debt " << (juce::int64) d.outputTimelineDropDebt
         << " excess " << juce::String(d.excessOutputQueueMs, 1) << " ms\n"
         << "Starved " << (juce::int64) d.outputStarvations
         << " late-drop " << (juce::int64) d.lateOutputFramesDropped
         << " trim " << (juce::int64) d.outputBacklogFramesTrimmed << "\n"
         << "FIFO in " << d.hostInputFifoDepth
         << " ovf " << (juce::int64) d.inputFifoOverflows << "/"
         << (juce::int64) d.outputFifoOverflows
         << " invalid/reuse " << (juce::int64) d.invalidGNNFrames << "/"
         << (juce::int64) d.reusedGNNFrames
         << "\nGNN active bins " << d.gnnActiveBinCount << "/257"
         << " pressure-hold " << (juce::int64) d.gnnPressureDeferredFrames
         << " geom-refresh " << (juce::int64) d.gnnGeometryForcedInferences;

    if (d.activeWorkerMode == 1)
        text << "\nHybrid GNN " << juce::String(100.0f * d.gnnBinFraction, 1)
             << "% / DEL " << juce::String(100.0f * (1.0f - d.gnnBinFraction), 1)
             << "%  raw sum " << juce::String::formatted("%.3e", d.meanRawGNNWeightSum);
    else if (d.activeWorkerMode == 0)
        text << "\nDelaunay all bins";

    diagnosticsLabel.setText(text, juce::dontSendNotification);
    if (haveNewWeightFrame)
        repaint(mapArea);
}

void GNNReconstructorAudioProcessorEditor::updatePropertiesPanel()
{
    if (selectedNode != nullptr)
    {
        propTitle.setText("Settings: " + selectedNode->name
                          + (audioProcessor.mapSceneLocked && !selectedNode->isListener ? " [LOCKED]" : ""),
                          juce::dontSendNotification);
        xEdit.setText(juce::String(selectedNode->x, 3), juce::dontSendNotification);
        yEdit.setText(juce::String(selectedNode->y, 3), juce::dontSendNotification);
        zEdit.setText(juce::String(selectedNode->z, 3), juce::dontSendNotification);

        const bool showYaw = !selectedNode->isListener;
        azLabel.setVisible(showYaw);
        azEdit.setVisible(showYaw);
        if (showYaw)
        {
            azLabel.setText("Mic yaw -> global (deg):", juce::dontSendNotification);
            azEdit.setText(juce::String(radiansToDegrees(selectedNode->azimuth), 2), juce::dontSendNotification);
        }

        const bool canEditPosition = !audioProcessor.mapSceneLocked || selectedNode->isListener;
        xEdit.setEditable(canEditPosition);
        yEdit.setEditable(canEditPosition);
        zEdit.setEditable(canEditPosition);
        azEdit.setEditable(showYaw && !audioProcessor.mapSceneLocked);
    }
    else
    {
        propTitle.setText("Node Settings (click a node)", juce::dontSendNotification);
        xEdit.setText("-", juce::dontSendNotification);
        yEdit.setText("-", juce::dontSendNotification);
        zEdit.setText("-", juce::dontSendNotification);
        xEdit.setEditable(false);
        yEdit.setEditable(false);
        zEdit.setEditable(false);
        azLabel.setVisible(false);
        azEdit.setVisible(false);
    }
}

void GNNReconstructorAudioProcessorEditor::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colours::darkgrey);
    g.setColour(juce::Colours::white.withAlpha(0.92f));
    g.setFont(10.5f);
    g.drawText(audioProcessor.getBuildIdentity(), 20, 1, 560, 17, juce::Justification::centredLeft);

    g.setColour(juce::Colours::black);
    g.fillRect(mapArea);

    const float ppm = 20.0f * zoomFactor;
    const float gridStep = chooseGridStep(ppm);
    const auto bottomLeft = pixelToMap({ (float) mapArea.getX(), (float) mapArea.getBottom() });
    const auto topRight = pixelToMap({ (float) mapArea.getRight(), (float) mapArea.getY() });

    g.saveState();
    g.reduceClipRegion(mapArea);

    g.setColour(juce::Colours::white.withAlpha(0.14f));
    const float firstX = std::ceil(bottomLeft.x / gridStep) * gridStep;
    const float firstY = std::ceil(bottomLeft.y / gridStep) * gridStep;
    for (float x = firstX; x <= topRight.x + 0.5f * gridStep; x += gridStep)
    {
        const float px = mapToPixel({ x, viewCenterY }).x;
        g.drawLine(px, (float) mapArea.getY(), px, (float) mapArea.getBottom(), 1.0f);
    }
    for (float y = firstY; y <= topRight.y + 0.5f * gridStep; y += gridStep)
    {
        const float py = mapToPixel({ viewCenterX, y }).y;
        g.drawLine((float) mapArea.getX(), py, (float) mapArea.getRight(), py, 1.0f);
    }

    const auto origin = mapToPixel({ 0.0f, 0.0f });
    g.setColour(juce::Colours::white.withAlpha(0.30f));
    if (origin.x >= mapArea.getX() && origin.x <= mapArea.getRight())
        g.drawLine(origin.x, (float) mapArea.getY(), origin.x, (float) mapArea.getBottom(), 1.35f);
    if (origin.y >= mapArea.getY() && origin.y <= mapArea.getBottom())
        g.drawLine((float) mapArea.getX(), origin.y, (float) mapArea.getRight(), origin.y, 1.35f);

    // Purely graphical annotations: they never enter the processor's spatial snapshot.
    for (const auto& annotation : audioProcessor.mapAnnotations)
    {
        g.setColour(juce::Colours::white.withAlpha(0.90f));
        if (annotation.type == MapAnnotation::Type::wallSegment && annotation.points.size() >= 2)
        {
            const auto a = mapToPixel(annotation.points[0]);
            const auto b = mapToPixel(annotation.points[1]);
            g.drawLine(a.x, a.y, b.x, b.y, 2.0f);
        }
        else if (annotation.type == MapAnnotation::Type::source && !annotation.points.empty())
        {
            const auto p = mapToPixel(annotation.points.front());
            const auto star = makeStar(p, 9.0f, 4.0f);
            g.fillPath(star);
        }
        else if (annotation.type == MapAnnotation::Type::dashedStroke && annotation.points.size() >= 2)
        {
            juce::Path path;
            const auto p0 = mapToPixel(annotation.points.front());
            path.startNewSubPath(p0);
            for (size_t i = 1; i < annotation.points.size(); ++i)
                path.lineTo(mapToPixel(annotation.points[i]));
            const float dashLengths[] { 8.0f, 6.0f };
            juce::Path dashed;
            juce::PathStrokeType(1.5f).createDashedStroke(dashed, path, dashLengths, 2);
            g.fillPath(dashed);
        }
    }

    if (drawingAnnotation)
    {
        g.setColour(juce::Colours::white.withAlpha(0.60f));
        if (draftAnnotation.type == MapAnnotation::Type::wallSegment && draftAnnotation.points.size() >= 2)
        {
            const auto a = mapToPixel(draftAnnotation.points[0]);
            const auto b = mapToPixel(draftAnnotation.points[1]);
            g.drawLine(a.x, a.y, b.x, b.y, 2.0f);
        }
        else if (draftAnnotation.type == MapAnnotation::Type::dashedStroke && draftAnnotation.points.size() >= 2)
        {
            juce::Path path;
            path.startNewSubPath(mapToPixel(draftAnnotation.points.front()));
            for (size_t i = 1; i < draftAnnotation.points.size(); ++i)
                path.lineTo(mapToPixel(draftAnnotation.points[i]));
            const float dashLengths[] { 8.0f, 6.0f };
            juce::Path dashed;
            juce::PathStrokeType(1.5f).createDashedStroke(dashed, path, dashLengths, 2);
            g.fillPath(dashed);
        }
    }

    if (weightSnapshot.frameCounter > 0)
    {
        const auto listenerP = mapToPixel({ audioProcessor.listener.x, audioProcessor.listener.y });
        const int activeWeightMics = juce::jmin(audioProcessor.getActiveMicCount(),
                                                (int) audioProcessor.micArrays.size());
        for (int m = 0; m < activeWeightMics; ++m)
        {
            const auto& mic = audioProcessor.micArrays[(size_t) m];
            const auto micP = mapToPixel({ mic.x, mic.y });
            const float rel = juce::jlimit(0.0f, 1.0f, weightSnapshot.relativeMean[(size_t) m]);
            const auto weightColour = weightSnapshot.gnnMode ? juce::Colours::aqua : juce::Colours::yellow;
            g.setColour(weightColour.withAlpha(0.14f + 0.74f * rel));
            g.drawLine(listenerP.x, listenerP.y, micP.x, micP.y, 1.0f + 8.0f * rel);

            const int pct = juce::roundToInt(100.0f * rel);
            const int lo = juce::roundToInt(100.0f * weightSnapshot.relativeLow[(size_t) m]);
            const int mi = juce::roundToInt(100.0f * weightSnapshot.relativeMid[(size_t) m]);
            const int hi = juce::roundToInt(100.0f * weightSnapshot.relativeHigh[(size_t) m]);
            juce::String label;
            if (weightSnapshot.gnnMode)
                label << "USE " << pct << "%  G "
                      << juce::roundToInt(100.0f * weightSnapshot.gnnRelativeMean[(size_t) m])
                      << "% D "
                      << juce::roundToInt(100.0f * weightSnapshot.delaunayRelativeMean[(size_t) m])
                      << "%\nUSED L/M/H " << lo << "/" << mi << "/" << hi << "%";
            else
                label << "DEL " << pct << "%\nUSED L/M/H " << lo << "/" << mi << "/" << hi << "%";
            g.setColour(weightColour.withAlpha(0.96f));
            g.setFont(9.5f);
            g.drawFittedText(label, (int) micP.x - 68, (int) micP.y + 13, 136, 30,
                             juce::Justification::centredTop, 2);
        }

        if (weightHistoryCount > 1)
        {
            const juce::Rectangle<float> box((float) mapArea.getX() + 9.0f,
                                             (float) mapArea.getY() + 9.0f,
                                             292.0f, 110.0f);
            g.setColour(juce::Colours::black.withAlpha(0.74f));
            g.fillRoundedRectangle(box, 5.0f);
            g.setColour(juce::Colours::white.withAlpha(0.92f));
            g.setFont(10.0f);
            g.drawText((weightSnapshot.gnnMode ? "GNN RELATIVE" : "Delaunay") + juce::String(" weights - history (~20 s)"),
                       box.toNearestInt().withHeight(18).reduced(6, 0),
                       juce::Justification::centredLeft);

            const std::array<juce::Colour, GNNReconstructorAudioProcessor::kNumMics> colours {
                juce::Colours::cyan, juce::Colours::yellow, juce::Colours::magenta,
                juce::Colours::limegreen, juce::Colours::orange, juce::Colours::cornflowerblue,
                juce::Colours::salmon, juce::Colours::turquoise, juce::Colours::violet,
                juce::Colours::lightgoldenrodyellow
            };
            const float x0 = box.getX() + 8.0f;
            const float y0 = box.getY() + 24.0f;
            const float w = box.getWidth() - 16.0f;
            const float h = box.getHeight() - 39.0f;
            g.setColour(juce::Colours::white.withAlpha(0.16f));
            g.drawLine(x0, y0 + h * 0.5f, x0 + w, y0 + h * 0.5f, 0.8f);

            for (int m = 0; m < audioProcessor.getActiveMicCount(); ++m)
            {
                juce::Path path;
                const int oldest = (weightHistoryWriteIndex - weightHistoryCount + kWeightHistoryLength)
                                 % kWeightHistoryLength;
                for (int n = 0; n < weightHistoryCount; ++n)
                {
                    const int index = (oldest + n) % kWeightHistoryLength;
                    const float v = juce::jlimit(0.0f, 1.0f, weightHistory[(size_t) m][(size_t) index]);
                    const float xx = x0 + w * (float) n / (float) (weightHistoryCount - 1);
                    const float yy = y0 + h * (1.0f - v);
                    if (n == 0) path.startNewSubPath(xx, yy); else path.lineTo(xx, yy);
                }
                g.setColour(colours[(size_t) m].withAlpha(0.92f));
                g.strokePath(path, juce::PathStrokeType(1.35f));
                g.setFont(8.5f);
                g.drawText("A" + juce::String(m + 1),
                           (int) (box.getX() + 8.0f + 27.0f * (float) m),
                           (int) box.getBottom() - 14, 30, 11,
                           juce::Justification::centredLeft);
            }
        }
    }

    {
        const juce::Rectangle<float> table((float) mapArea.getRight() - 304.0f,
                                           (float) mapArea.getY() + 9.0f,
                                           294.0f, 258.0f);
        g.setColour(juce::Colours::black.withAlpha(0.84f));
        g.fillRoundedRectangle(table, 5.0f);
        g.setColour(juce::Colours::white.withAlpha(0.95f));
        g.setFont(10.0f);
        const juce::String modeName = weightSnapshot.frameCounter == 0 ? "WAITING"
                                    : (weightSnapshot.gnnMode ? "HYBRID GNN/DEL" : "DELAUNAY");
        g.drawText("NODE WEIGHTS - " + modeName,
                   table.toNearestInt().withHeight(18).reduced(6, 0),
                   juce::Justification::centredLeft);
        g.setFont(8.8f);
        g.setColour(juce::Colours::white.withAlpha(0.72f));
        g.drawText(weightSnapshot.gnnMode ? "U=used   G=GNN(relative)   D=Delaunay" : "U=used Delaunay",
                   (int) table.getX() + 7, (int) table.getY() + 17,
                   (int) table.getWidth() - 14, 14, juce::Justification::centredLeft);
        g.setFont(9.6f);
        g.setColour(juce::Colours::white.withAlpha(0.95f));
        for (int m = 0; m < audioProcessor.getActiveMicCount(); ++m)
        {
            juce::String row;
            if (weightSnapshot.frameCounter > 0)
            {
                row << "A" << (m + 1)
                    << "  U " << juce::String(100.0f * weightSnapshot.relativeMean[(size_t) m], 1) << "%";
                if (weightSnapshot.gnnMode)
                    row << "   G " << juce::String(100.0f * weightSnapshot.gnnRelativeMean[(size_t) m], 1) << "%"
                        << "   D " << juce::String(100.0f * weightSnapshot.delaunayRelativeMean[(size_t) m], 1) << "%";
            }
            else
                row << "A" << (m + 1) << "  --";
            g.drawText(row, (int) table.getX() + 7, (int) table.getY() + 33 + 18 * m,
                       (int) table.getWidth() - 14, 18, juce::Justification::centredLeft);
        }
        if (weightSnapshot.frameCounter > 0)
        {
            juce::String sumRow;
            if (weightSnapshot.gnnMode)
                sumRow << "GNN bins " << juce::String(100.0f * weightSnapshot.gnnBinFraction, 1)
                       << "%  U sum " << juce::String(weightSnapshot.meanAppliedSum, 3)
                       << "  raw " << juce::String::formatted("%.2e", weightSnapshot.meanRawSum);
            else
                sumRow << "Applied sum " << juce::String(weightSnapshot.meanAppliedSum, 3);
            g.setColour(juce::Colours::white.withAlpha(0.78f));
            g.setFont(8.8f);
            g.drawText(sumRow, (int) table.getX() + 7, (int) table.getBottom() - 17,
                       (int) table.getWidth() - 14, 14, juce::Justification::centredLeft);
        }
    }

    auto drawNode = [&](SpatialNode& node, juce::Colour colour)
    {
        const auto p = mapToPixel({ node.x, node.y });
        if (selectedNode == &node)
        {
            g.setColour(juce::Colours::white);
            g.drawEllipse(p.x - 14.0f, p.y - 14.0f, 28.0f, 28.0f, 2.0f);
        }
        g.setColour(colour);
        g.fillEllipse(p.x - 10.0f, p.y - 10.0f, 20.0f, 20.0f);

        juce::String label = node.name + "  z=" + juce::String(node.z, 2);
        if (!node.isListener)
        {
            g.setColour(juce::Colours::white);
            g.drawLine(p.x, p.y,
                       p.x + 15.0f * std::cos(node.azimuth),
                       p.y - 15.0f * std::sin(node.azimuth), 2.0f);
            label << "  yaw=" << juce::String(radiansToDegrees(node.azimuth), 1) << juce::String::fromUTF8("\xC2\xB0");
        }
        g.setColour(juce::Colours::white);
        g.drawText(label, (int) p.x - 75, (int) p.y - 34, 150, 20, juce::Justification::centred);
    };

    const int activeMics = juce::jmin(audioProcessor.getSceneMicCount(),
                                      (int) audioProcessor.micArrays.size());
    for (int m = 0; m < activeMics; ++m)
        drawNode(audioProcessor.micArrays[(size_t) m], juce::Colours::cornflowerblue);
    drawNode(audioProcessor.listener, juce::Colours::orange);

    {
        const juce::Rectangle<float> walkBox((float) mapArea.getX() + 115.0f,
                                             (float) mapArea.getBottom() - 34.0f,
                                             (float) mapArea.getWidth() - 230.0f, 25.0f);
        g.setColour(juce::Colours::black.withAlpha(0.76f));
        g.fillRoundedRectangle(walkBox, 5.0f);
        const float walkSpeed = std::hypot(walkVelocityX, walkVelocityY);
        g.setColour(walkSpeed > 0.01f ? juce::Colours::orange : juce::Colours::white.withAlpha(0.88f));
        g.setFont(9.5f);
        const juce::String walkText = walkSpeed > 0.01f
            ? "Walking " + juce::String(walkSpeed, 2) + " m/s  |  release = smooth stop"
            : "ARROWS: walk query  1.35 m/s  |  Shift: 0.35 m/s";
        g.drawText(walkText, walkBox.toNearestInt().reduced(6, 0), juce::Justification::centred);
    }

    g.restoreState();

    // Global-map compass: this also documents the yaw convention in degrees.
    {
        const float x = (float) mapArea.getX() + 44.0f;
        const float y = (float) mapArea.getBottom() - 45.0f;
        g.setColour(juce::Colours::black.withAlpha(0.72f));
        g.fillRoundedRectangle(x - 33.0f, y - 34.0f, 92.0f, 68.0f, 5.0f);
        g.setColour(juce::Colours::white.withAlpha(0.95f));
        g.drawLine(x, y, x + 28.0f, y, 1.8f);
        g.drawLine(x, y, x, y - 28.0f, 1.8f);
        g.drawLine(x + 28.0f, y, x + 22.0f, y - 4.0f, 1.8f);
        g.drawLine(x + 28.0f, y, x + 22.0f, y + 4.0f, 1.8f);
        g.drawLine(x, y - 28.0f, x - 4.0f, y - 22.0f, 1.8f);
        g.drawLine(x, y - 28.0f, x + 4.0f, y - 22.0f, 1.8f);
        g.setFont(9.0f);
        g.drawText("+X 0" + juce::String::fromUTF8("\xC2\xB0"), (int) x + 29, (int) y - 8, 48, 15, juce::Justification::left);
        g.drawText("+Y 90" + juce::String::fromUTF8("\xC2\xB0"), (int) x - 20, (int) y - 45, 55, 15, juce::Justification::centred);
        g.drawText("GLOBAL", (int) x - 28, (int) y + 10, 58, 13, juce::Justification::centred);
    }

    g.setColour(juce::Colours::white);
    g.drawText("Interpolation:", 600, 14, 350, 20, juce::Justification::left);
    g.drawText("FOA nodes:", 600, 68, 350, 20, juce::Justification::left);
    g.drawText("GNN threshold (below = Delaunay, dB rel.):", 600, 122, 350, 20, juce::Justification::left);
    g.drawText("DirAC covariance smoothing:", 600, 176, 350, 20, juce::Justification::left);
    g.drawText("GNN temporal average (experimental):", 600, 230, 350, 20, juce::Justification::left);

    g.setColour(juce::Colours::white.withAlpha(0.80f));
    g.setFont(11.5f);
    g.drawFittedText("Manual I/O: choose N=1..10 here. The plug-in keeps one fixed 40-channel input bus and consumes only the first 4*N channels (FOA order W,Y,Z,X); no host auto-detection or pin remapping occurs during playback. The checkpoint was trained with N=5: GNN at N!=5 is an OOD graph extrapolation with training-degree compensation. Delaunay remains geometry-defined for every supported N.",
                     600, 284, 350, 52, juce::Justification::topLeft, 4);
    g.drawFittedText("Map tools are display-only: click the map, then Arrow keys walk the orange query at 1.35 m/s; hold Shift for 0.35 m/s precision. Fit View recentres/scales any coordinate set. Lock Scene freezes known arrays + annotations but keyboard/drag still move the orange query. Mic yaw is active local->global; the orange node has no orientation.",
                     600, 650, 350, 72, juce::Justification::topLeft, 5);

    g.setColour(audioProcessor.mapSceneLocked ? juce::Colours::lightgreen : juce::Colours::white.withAlpha(0.65f));
    g.setFont(10.0f);
    const juce::String toolName = currentTool == MapTool::pan ? "HAND"
                                : currentTool == MapTool::moveNode ? "NODE"
                                : currentTool == MapTool::wall ? "WALL"
                                : currentTool == MapTool::source ? "SOURCE" : "DASHED";
    g.drawText(audioProcessor.mapSceneLocked ? "MAP LOCKED - orange query moves by drag or Arrow keys"
                                             : "Map tool: " + toolName + "  |  Hand pans view; Shift-drag a blue mic in Node mode rotates it",
               20, 741, 560, 17, juce::Justification::centredLeft);
}

void GNNReconstructorAudioProcessorEditor::resized()
{
    mapArea = { 20, 58, 560, 680 };

    int bx = 20;
    const int by = 24, bh = 28, gap = 4;
    fitViewButton.setBounds(bx, by, 62, bh); bx += 62 + gap;
    panToolButton.setBounds(bx, by, 56, bh); bx += 56 + gap;
    moveToolButton.setBounds(bx, by, 54, bh); bx += 54 + gap;
    wallToolButton.setBounds(bx, by, 54, bh); bx += 54 + gap;
    sourceToolButton.setBounds(bx, by, 62, bh); bx += 62 + gap;
    dashedToolButton.setBounds(bx, by, 62, bh); bx += 62 + gap;
    undoAnnotationButton.setBounds(bx, by, 50, bh); bx += 50 + gap;
    lockSceneButton.setBounds(bx, by, 86, bh);

    interpSelector.setBounds(600, 36, 350, 28);
    nodeCountSelector.setBounds(600, 90, 350, 28);
    thresholdSlider.setBounds(600, 144, 350, 28);
    smoothingSlider.setBounds(600, 198, 350, 28);
    temporalWindowSlider.setBounds(600, 252, 350, 28);

    propTitle.setBounds(600, 338, 350, 24);
    xLabel.setBounds(600, 366, 125, 22); xEdit.setBounds(730, 366, 220, 22);
    yLabel.setBounds(600, 392, 125, 22); yEdit.setBounds(730, 392, 220, 22);
    zLabel.setBounds(600, 418, 125, 22); zEdit.setBounds(730, 418, 220, 22);
    azLabel.setBounds(600, 444, 125, 22); azEdit.setBounds(730, 444, 220, 22);
    diagnosticsLabel.setBounds(600, 472, 350, 174);
}

SpatialNode* GNNReconstructorAudioProcessorEditor::getNodeAtPosition(juce::Point<float> pos)
{
    const auto listenerP = mapToPixel({ audioProcessor.listener.x, audioProcessor.listener.y });
    if (listenerP.getDistanceFrom(pos) < 20.0f)
        return &audioProcessor.listener;

    if (audioProcessor.mapSceneLocked)
        return nullptr;

    const int activeMics = juce::jmin(audioProcessor.getSceneMicCount(),
                                      (int) audioProcessor.micArrays.size());
    for (int m = 0; m < activeMics; ++m)
    {
        auto& mic = audioProcessor.micArrays[(size_t) m];
        if (mapToPixel({ mic.x, mic.y }).getDistanceFrom(pos) < 20.0f)
            return &mic;
    }
    return nullptr;
}

void GNNReconstructorAudioProcessorEditor::mouseDown(const juce::MouseEvent& e)
{
    if (!mapArea.contains(e.getPosition()))
        return;

    // A map click opts into walking control without stealing arrows from the
    // property fields or host UI before the user interacts with the map.
    grabKeyboardFocus();

    if (currentTool == MapTool::pan)
    {
        panningView = true;
        panStartCenterX = viewCenterX;
        panStartCenterY = viewCenterY;
        draggedNode = nullptr;
        return;
    }

    if (currentTool == MapTool::moveNode)
    {
        draggedNode = getNodeAtPosition(e.position);
        if (draggedNode != nullptr)
        {
            selectedNode = draggedNode;
            dragStartAzimuth = draggedNode->azimuth;
            updatePropertiesPanel();
        }
        repaint();
        return;
    }

    if (audioProcessor.mapSceneLocked)
        return;

    const auto mapPoint = pixelToMap(e.position);
    if (currentTool == MapTool::source)
    {
        MapAnnotation annotation;
        annotation.type = MapAnnotation::Type::source;
        annotation.points.push_back(mapPoint);
        commitAnnotation(std::move(annotation));
        return;
    }

    drawingAnnotation = true;
    draftAnnotation = {};
    draftAnnotation.type = currentTool == MapTool::wall ? MapAnnotation::Type::wallSegment
                                                        : MapAnnotation::Type::dashedStroke;
    draftAnnotation.points.push_back(mapPoint);
    if (draftAnnotation.type == MapAnnotation::Type::wallSegment)
        draftAnnotation.points.push_back(mapPoint);
    repaint(mapArea);
}

void GNNReconstructorAudioProcessorEditor::mouseDrag(const juce::MouseEvent& e)
{
    if (panningView)
    {
        const auto delta = e.getOffsetFromDragStart();
        const float ppm = 20.0f * zoomFactor;
        viewCenterX = panStartCenterX - (float) delta.x / ppm;
        viewCenterY = panStartCenterY + (float) delta.y / ppm;
        storeMapViewState();
        repaint(mapArea);
        return;
    }

    if (draggedNode != nullptr && mapArea.contains(e.getPosition()))
    {
        if (e.mods.isShiftDown() && !draggedNode->isListener && !audioProcessor.mapSceneLocked)
        {
            const float deltaDegrees = (float) e.getOffsetFromDragStart().x * 0.5f;
            draggedNode->azimuth = dragStartAzimuth + degreesToRadians(deltaDegrees);
        }
        else
        {
            const auto mapPoint = pixelToMap(e.position);
            draggedNode->x = mapPoint.x;
            draggedNode->y = mapPoint.y;
        }
        publishSpatialEdit();
        updatePropertiesPanel();
        repaint();
        return;
    }

    if (!drawingAnnotation || audioProcessor.mapSceneLocked || !mapArea.contains(e.getPosition()))
        return;

    const auto p = pixelToMap(e.position);
    if (draftAnnotation.type == MapAnnotation::Type::wallSegment)
    {
        if (draftAnnotation.points.size() < 2) draftAnnotation.points.push_back(p);
        else draftAnnotation.points[1] = p;
    }
    else if (draftAnnotation.type == MapAnnotation::Type::dashedStroke)
    {
        const auto lastPx = mapToPixel(draftAnnotation.points.back());
        if (lastPx.getDistanceFrom(e.position) >= 8.0f)
            draftAnnotation.points.push_back(p);
    }
    repaint(mapArea);
}

void GNNReconstructorAudioProcessorEditor::mouseUp(const juce::MouseEvent& e)
{
    juce::ignoreUnused(e);
    if (panningView)
    {
        panningView = false;
        storeMapViewState();
    }
    draggedNode = nullptr;

    if (!drawingAnnotation)
        return;

    drawingAnnotation = false;
    bool valid = false;
    if (draftAnnotation.type == MapAnnotation::Type::wallSegment && draftAnnotation.points.size() >= 2)
    {
        const auto a = mapToPixel(draftAnnotation.points[0]);
        const auto b = mapToPixel(draftAnnotation.points[1]);
        valid = a.getDistanceFrom(b) >= 5.0f;
    }
    else if (draftAnnotation.type == MapAnnotation::Type::dashedStroke)
        valid = draftAnnotation.points.size() >= 2;

    if (valid)
        commitAnnotation(std::move(draftAnnotation));
    draftAnnotation = {};
    repaint(mapArea);
}

void GNNReconstructorAudioProcessorEditor::mouseWheelMove(
    const juce::MouseEvent& e,
    const juce::MouseWheelDetails& wheel)
{
    if (!mapArea.contains(e.position.toInt()))
        return;

    const auto anchorMap = pixelToMap(e.position);
    const float scale = std::exp(wheel.deltaY * 0.6f);
    zoomFactor = juce::jlimit(0.01f, 20.0f, zoomFactor * scale);
    const float ppm = 20.0f * zoomFactor;
    viewCenterX = anchorMap.x - (e.position.x - (float) mapArea.getCentreX()) / ppm;
    viewCenterY = anchorMap.y + (e.position.y - (float) mapArea.getCentreY()) / ppm;
    storeMapViewState();
    repaint(mapArea);
}
