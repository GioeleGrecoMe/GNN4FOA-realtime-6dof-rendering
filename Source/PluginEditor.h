// GNN4FOA v2
// Authors: Gioele Greco, Filippo Marri, Mirco Pezzoli, Fabio Antonacci
// Image and Sound Processing Lab (ISPL), Politecnico di Milano
// Copyright (c) 2026 Gioele Greco, Filippo Marri, Mirco Pezzoli, Fabio Antonacci

#pragma once

#include <JuceHeader.h>
#include "PluginProcessor.h"

class GNNReconstructorAudioProcessorEditor : public juce::AudioProcessorEditor,
                                              private juce::Timer
{
public:
    explicit GNNReconstructorAudioProcessorEditor(GNNReconstructorAudioProcessor&);
    ~GNNReconstructorAudioProcessorEditor() override;

    void paint(juce::Graphics&) override;
    void resized() override;
    void mouseDown(const juce::MouseEvent& e) override;
    void mouseDrag(const juce::MouseEvent& e) override;
    void mouseUp(const juce::MouseEvent& e) override;
    void mouseWheelMove(const juce::MouseEvent& e, const juce::MouseWheelDetails& wheel) override;
    bool keyPressed(const juce::KeyPress& key) override;
    bool keyStateChanged(bool isKeyDown) override;

private:
    enum class MapTool { pan, moveNode, wall, source, dashed };

    GNNReconstructorAudioProcessor& audioProcessor;

    juce::ComboBox interpSelector;
    juce::ComboBox nodeCountSelector;
    juce::Slider thresholdSlider;
    juce::Slider smoothingSlider;
    juce::Slider temporalWindowSlider;

    std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment> interpAttach;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment> nodeCountAttach;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> threshAttach;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> smoothAttach;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> temporalWindowAttach;

    juce::TextButton fitViewButton { "Fit View" };
    juce::TextButton panToolButton { "Hand" };
    juce::TextButton moveToolButton { "Node" };
    juce::TextButton wallToolButton { "Wall" };
    juce::TextButton sourceToolButton { "Source" };
    juce::TextButton dashedToolButton { "Dashed" };
    juce::TextButton undoAnnotationButton { "Undo" };
    juce::TextButton lockSceneButton { "Lock Scene" };

    juce::Label propTitle;
    juce::Label xLabel, yLabel, zLabel, azLabel;
    juce::Label xEdit, yEdit, zEdit, azEdit;
    juce::Label diagnosticsLabel;

    SpatialNode* selectedNode = nullptr;
    SpatialNode* draggedNode = nullptr;
    juce::Rectangle<int> mapArea;
    float zoomFactor = 1.0f;
    float viewCenterX = 0.0f;
    float viewCenterY = 0.0f;
    float dragStartAzimuth = 0.0f;
    MapTool currentTool = MapTool::pan;
    bool drawingAnnotation = false;
    bool panningView = false;
    float panStartCenterX = 0.0f;
    float panStartCenterY = 0.0f;
    MapAnnotation draftAnnotation;
    DiracGNNEngine::WeightVisualizationSnapshot weightSnapshot;

    static constexpr int kWeightHistoryLength = 80; // ~20 s at 4 Hz
    std::array<std::array<float, kWeightHistoryLength>, GNNReconstructorAudioProcessor::kNumMics> weightHistory {};
    int weightHistoryWriteIndex = 0;
    int weightHistoryCount = 0;
    uint64_t lastWeightHistoryFrameCounter = 0;
    int lastWeightHistoryMode = -1;

    // Human-scale keyboard locomotion for the reconstructed/query node.
    // Kept entirely on the message thread; DSP receives only the resulting
    // lock-free spatial snapshots.
    float walkVelocityX = 0.0f;
    float walkVelocityY = 0.0f;
    double lastWalkTickMs = 0.0;
    double lastDiagnosticsUpdateMs = 0.0;
    bool walkHighRateTimer = false;
    int walkPropertyRefreshDivider = 0;
    static constexpr float kWalkSpeedMetersPerSecond = 1.35f;
    static constexpr float kPrecisionWalkSpeedMetersPerSecond = 0.35f;
    static constexpr int kWalkTimerHz = 50;

    void timerCallback() override;
    void updatePropertiesPanel();
    void updateToolState();
    void setMapTool(MapTool tool);
    void fitMapView();
    void storeMapViewState() noexcept;
    void commitAnnotation(MapAnnotation annotation);
    SpatialNode* getNodeAtPosition(juce::Point<float> pos);
    juce::Point<float> mapToPixel(juce::Point<float> mapPoint) const noexcept;
    juce::Point<float> pixelToMap(juce::Point<float> pixelPoint) const noexcept;
    void publishSpatialEdit() noexcept;
    bool shouldHandleWalkingKeys() const noexcept;
    bool anyWalkingArrowDown() const noexcept;
    void updateKeyboardWalking(double nowMs);
    void setWalkingTimerHighRate(bool highRate);
    void softlyKeepListenerVisible();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(GNNReconstructorAudioProcessorEditor)
};
