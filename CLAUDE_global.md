# Code Style: Scientific Computing Standards
## Readability
- MUST write functions that do one thing only — keep cognitive load minimal
- MUST use consistent, meaningful names — no `a`, `foo`, `results2`
- MUST use consistent formatting and naming conventions throughout (no mixing styles)

## Don't Repeat Yourself
- Every constant or data value MUST have a single authoritative definition
- Re-use existing libraries for well-established problems (e.g. numerical integration, matrix ops)

## Correctness First
- Add assertions to validate inputs, outputs, and intermediate states
- NEVER optimize for performance before the code is correct and tested

## Documentation
- Document WHY and WHAT (interfaces, intent, design decisions) — not HOW
- NEVER write comments that merely restate the code (e.g. `i = i + 1 # increment i`)
- If a block of code needs a long explanation, refactor it instead
- Embed documentation in the code (docstrings, not separate files)

## Automation & Reproducibility
- Repetitive tasks MUST be scripted — never rely on manual re-execution
- Record all parameters, library versions, and data identifiers used to generate outputs
- Use a build tool or workflow manager to express dependencies between steps
