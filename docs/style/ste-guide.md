# STE100-inspired writing guide

This guide defines the house writing style for PACT descriptive documentation.
It is inspired by ASD-STE100 Simplified Technical English. It is not a full
licensed STE100 dictionary.

## Goals

- A new reader can learn what a unit does without reading the source first.
- Sentences stay short, clear, and consistent.
- Pages state facts about the current system. They do not argue for a design.

## Sentence rules

1. Use the present tense for system behavior.
2. Use the active voice when the actor is known.
3. Put one idea in each sentence.
4. Keep descriptive sentences at or under 25 words when you can.
5. Do not drop articles (`a`, `an`, `the`).
6. Prefer simple verbs from everyday English plus the technical names list.
7. Avoid stacked noun phrases. Prefer a short clause.

## Section rules

1. Follow the module or directory template section order.
2. Use numbered steps for Behavior when order matters.
3. Write `None.` when a section has no content.
4. Link only to related descriptive pages. Do not link to ADR files.

## Banned rationale language

Do not use these words or phrases on descriptive pages:

- because
- in order to
- rather than
- instead of
- we chose
- the reason
- so that
- to allow
- to enable
- designed to
- intended to

If a sentence needs one of these phrases, move the content to an ADR.

## ADR reference ban

Descriptive pages, source comments, and code docstrings must not cite ADR
identifiers or ADR paths. Only ADR files and ADR index pages may cite ADRs.

## Stub pages

When the source unit is a placeholder, say so in Purpose and Constraints.
Keep sections short. Expand the page when the unit is no longer a stub.

## Typography

- Use ASCII punctuation in new docs.
- Wrap lines near 100 columns when practical.
- Use fenced code only for short type or path examples.
