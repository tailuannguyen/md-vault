# Vault navigation and agent instructions

## Start here

- [README](README.md): the vault's purpose and organization.
- [Natural language processing](nlp.md): course overview and NLP lecture outline.
- [GPU architecture and programming](gpu_architecture.md): course overview and GPU lecture outline.

The subject overviews link to their lecture notes and derivation companions. The instructions below explain how to maintain this vault.

## Purpose and scope

This repository is a personal Markdown vault holding the owner's context, knowledge, and the connections within that knowledge. It supports personal review and provides useful context for agents. It is used with Obsidian and should remain easy to navigate in a terminal.

The scope extends beyond coursework to professional work, personal projects, hobbies, and other areas of life. Let the organization grow with the material the owner adds; do not assume the whole vault should follow a course structure.

## Note size and organization

- Prefer a small number of substantial, cohesive notes over many tiny topic files.
- For coursework, default to one note per lecture. Use headings within that note for concepts, examples, and supporting details.
- Keep lecture topics together unless multiple major concepts benefit from separate, substantial notes. Put lengthy foundational derivations and proofs in substantial companion notes, linked directly from the relevant lecture section.
- Apply the same principle elsewhere: choose a meaningful unit such as a project or broad subject, and keep related details together.
- Do not create a separate file for every definition, formula, technique, or minor subtopic. Existing fine-grained notes are not a template to reproduce.
- Keep notes focused and useful for review; preferring larger units does not mean adding unnecessary detail.
- For lecture PDF conversions, deliver the lecture Markdown notes and substantial derivation notes where needed, and maintain the navigation hierarchy below. Do not add review guides, generated quizzes, separate index notes, source/coverage reports, graph files, relationship proposals, or JSON context exports unless newly requested.
- Prefer plain headings, prose, equations, and useful examples. Avoid unnecessary frontmatter and decorative callouts.

## Lecture depth, derivations, and quality

- The preference for fewer files concerns organization, not a strict word limit. A short synopsis or list of topic names is insufficient when the lecture teaches substantial methods or mathematics.
- Preserve important explanations, training objectives, method steps, foundational derivations, proofs, assumptions, and representative examples. The notes and their linked companions should retain enough information to understand the lecture after the PDFs are deleted.
- Apply this depth standard during the initial conversion of new material. Do not wait for the owner to request a quality check or point out missing mathematics before preserving it.
- Foundational mathematics is important to the owner. Define notation, state assumptions, and show the intermediate reasoning needed to reproduce a derivation or proof; do not retain only the final formula.
- Keep brief derivations, short calculations, and representative worked examples inside the lecture. Group closely related lengthy derivations in one substantial companion note rather than creating a file for every equation or calculation. Create these companions as part of the conversion without needing a separate request.
- Keep the problem setup, main results, and their interpretation in the lecture. Place a direct link to the relevant companion heading in the section where the result is introduced, and include a return link from the companion to the lecture. A generic further-reading link at the end is insufficient.
- A derivation companion must explain its notation and assumptions, show intermediate algebra or chain-rule steps, justify its conclusion, and address relevant boundary cases. A worked calculation should show how to reach the answer rather than only stating it.
- Match depth to the material: explain central methods and derivations thoroughly, while keeping survey topics proportionate. Repeated slides and incidental listings may be condensed without losing the lecture's substantive lessons.
- Before completing a conversion, compare the resulting lecture and companions with the source for correctness, substantive coverage, and explanatory depth. Correct final formulas alone do not establish that a conversion is complete. Preserve source-taught proofs and the reasoning needed to reproduce central methods; keep survey mentions and explicitly skipped material proportionate.
- Check mathematical signs, dimensions, assumptions, limiting cases, and worked arithmetic; use numerical gradient or other computational checks where useful. Correct source mistakes rather than copying them into the notes.
- Verify new and affected links, including companion heading anchors and return links. Include every companion in its subject overview and update the root README as required below. Keep verification findings in the task response rather than adding source/coverage reports to the vault.

## Vault navigation

- Use the root `AGENTS.md` as both the central navigation node and the agent instruction file. Keep links to `README.md` and each subject or area overview in its opening navigation section. Do not recreate a separate index note.
- Keep broad subject overview notes at the root, such as `nlp.md` and `gpu_architecture.md`. For a course, include its overview, topics, educational goals, and a concise linked outline of every note belonging to it, including substantial derivation companions.
- Follow the hierarchy: `AGENTS.md` → subject overview → lecture or other substantive notes. The README introduces the vault and links to the subject overviews; detailed note listings belong in those overviews.
- When adding or converting material, update the relevant subject outline and the root README in the same task. Add a navigation link in `AGENTS.md` when creating a new subject or area overview. Preserve the owner's supplied course descriptions and do not create empty nodes for syllabus topics without notes.

## Links and knowledge graphs

- Keep initial linking sparse and intentional. Include links for the navigation hierarchy, major concept notes split from a parent note, or its derivation/proof companion notes.
- The owner will evaluate broader connections later. Do not proactively add numerous links between lectures, topics, courses, projects, or life areas merely because they appear related.
- Do not infer speculative relationships or build a dense knowledge graph by default. Add broader connections when the owner requests or identifies them.
- When a graph is requested, favor broad, meaningful concepts as nodes rather than every small topic. Follow the requested scope without expanding it across the vault.

## File conventions and editing

- Use descriptive filenames without spaces. Use lowercase snake_case for content Markdown files and directories, including acronyms (for example, `gpu_memory_and_performance.md`); retain conventional root filenames `AGENTS.md` and `README.md`.
- For numbered lecture notes, start the filename with the zero-padded number (and suffix where applicable), followed by the topic; omit the `lecture_` prefix. Examples: `06_word2vec.md` and `01a_history_and_representations.md`.
- Use Obsidian-compatible Markdown and working internal links while keeping the text readable outside Obsidian.
- Lecture notes must stand alone after the source PDFs are deleted. Omit PDF citations, links, page references, source metadata, and coverage sections. Do not delete the original PDFs unless requested; the owner may remove them later.
- Do not invent personal context, experiences, or preferences for the owner.
- Apply these guidelines to the work requested. Do not reorganize existing notes, rename unrelated files, or rewrite existing graphs solely to enforce these instructions.
- Preserve the owner's edits and unrelated repository changes.

## Root README maintenance

- Read the root `README.md` before converting any new material into Markdown, alongside these instructions, to understand the vault's current organization.
- Every task that converts new material into Markdown must also update the root `README.md` in the same task, whether it creates new notes or adds the material to existing notes. This is required even when the owner does not explicitly ask for a README update. The conversion is not complete until the README reflects the resulting notes.
- Add or adjust concise descriptions and working links under the appropriate subject or area so the new material is easy to find; add a new area when the material warrants it. Verify affected README links before finishing.
- Keep the README a brief overview and navigation entry point. Do not turn it into a review guide, source/coverage report, or detailed knowledge graph, and do not list areas that have no notes yet.
- Reflect the notes that actually exist, preserve unrelated entries, and keep navigation links current when notes are renamed or reorganized within the requested work.
