JUCE is not vendored in this repository. BUILD_UNIVERSAL.command uses
ThirdParty/JUCE when present, uses JUCE_ROOT when supplied, or clones the exact
JUCE 9.0.1 tag into ThirdParty/JUCE.

The Eigen 5.0.0 headers required by the embedded GNN runtime are vendored under
Source/Eigen. Third-party license information is collected in
THIRD_PARTY_NOTICES.md and ThirdParty/licenses/.
