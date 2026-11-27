@testable import MferenceAppCore
import Testing

/// The picker's two groupings: every model listed exactly once, Apple's row
/// only while it is offered, and no empty sections.
@Suite struct ModelPickerGroupingTests {
    @Test(arguments: ModelPickerGrouping.allCases)
    func everyModelIsListedExactlyOnce(_ grouping: ModelPickerGrouping) {
        let listed = grouping.sections(for: AppModel.allCases).flatMap(\.models)
        #expect(listed.count == AppModel.allCases.count)
        #expect(Set(listed) == Set(AppModel.allCases))
    }

    @Test func bySizeIsSmallLargeThenMixtureOfExperts() {
        let sections = ModelPickerGrouping.size.sections(for: AppModel.allCases)
        #expect(sections.map(\.title) == ["Small", "Large", "Mixture of experts"])
        #expect(sections.map(\.models) == [[.quick, .minicpm5, .minicpm5_1b, .apple],
                                           [.deep, .bonsai27b],
                                           [.qwen36, .inkling2Mini, .maple]])
        // Inkling keeps much of itself resident: no SSD claim on its section.
        #expect(sections[2].hint == nil)
    }

    @Test func byFamilyGroupsTheQwenModelsTogether() {
        let sections = ModelPickerGrouping.family.sections(for: AppModel.allCases)
        #expect(sections.map(\.title) == ["Qwen", "MiniCPM", "Inkling", "Bonsai", "Maple", "Apple"])
        #expect(sections[0].models == [.quick, .deep, .qwen36])
        #expect(sections[1].models == [.minicpm5, .minicpm5_1b])
    }

    @Test func appleIsLeftOutWhileNotOfferedAndItsEmptySectionDropped() {
        let offered = AppModel.engineModels
        #expect(!ModelPickerGrouping.size.sections(for: offered).flatMap(\.models).contains(.apple))
        #expect(!ModelPickerGrouping.family.sections(for: offered).map(\.title).contains("Apple"))
    }

    @Test func recommendedStaysQuickAndDeep() {
        #expect(AppModel.allCases.filter { $0.group == .recommended } == [.quick, .deep])
        #expect(ModelPickerGrouping(rawValue: "size") == .size)
        #expect(ModelPickerGrouping.allCases.map(\.title) == ["By size", "By family"])
    }
}
