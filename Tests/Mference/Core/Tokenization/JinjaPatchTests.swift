import Jinja
import Testing

/// The two local swift-jinja patches (`Vendor/swift-jinja`): literal braces keep
/// their whitespace-control behaviour, and the `nullOutput` policy renders
/// `none` as Python does while leaving null tests and omitted `else` alone.
@Suite("Jinja local patches: null output and literal-brace whitespace")
struct JinjaPatchTests {
    @Test func pythonNullOutputIsScopedAndPreservesNullTests() throws {
        let template = try Template("{% macro value(x) %}{{ x }}{% endmacro %}{{ value(none) }}|{{ missing }}|{{ none is none }}")
        let environment = Environment()
        environment.policies.nullOutput = "None"
        #expect(try template.render([:], environment: environment) == "None||true")
        #expect(try template.render([:]) == "||true")
        let conditional = try Template("{{ 'x' if false }}|{{ 'x' if false else none }}|{{ ('x' if false) is undefined }}")
        #expect(try conditional.render([:], environment: environment) == "|None|true")
        #expect(try conditional.render([:]) == "||true")
    }

    @Test func literalBracesDoNotAcquireProtectiveSpaces() throws {
        let source = "parameters:{\n{%- if true -%}properties:{ {{- 'value' -}} }{%- endif -%}"
        #expect(try Template(source).render([:]) == "parameters:{properties:{value}")
        #expect(try Template("a{ \n{#- note -#}\n b").render([:]) == "a{b")
    }

    @Test func tojsonTakesSeparatorsAndSortKeys() throws {
        // transformers' tojson passes both through to json.dumps; Inkling 2's
        // template uses them for tool declarations and tool-call arguments.
        let v = #"{%- set v = {"b": 1, "a": [1, "é"]} -%}"#
        func render(_ expression: String) throws -> String {
            try Template(v + "{{ v | " + expression + " }}").render([:])
        }
        #expect(try render(#"tojson(sort_keys=true, separators=(",", ":"))"#) == #"{"a":[1,"é"],"b":1}"#)
        #expect(try render(#"tojson(separators=(",", ":"))"#) == #"{"b":1,"a":[1,"é"]}"#)
        #expect(try render("tojson(sort_keys=true)") == #"{"a": [1, "é"], "b": 1}"#)
        // Without the new arguments nothing changes.
        #expect(try render("tojson") == #"{"b": 1, "a": [1, "é"]}"#)
        #expect(throws: (any Error).self) { try render(#"tojson(separators=(",",))"#) }
    }

    @Test func quotedDelimitersAreData() throws {
        #expect(try Template("{{ 'a -}} b' }}").render([:]) == "a -}} b")
        #expect(try Template("{{ 'a {%- b' }}").render([:]) == "a {%- b")
    }
}
