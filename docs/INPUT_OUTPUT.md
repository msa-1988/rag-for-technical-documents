# Input and Output Definition

## Main Input

### Document input

The app will accept:
- PDF files

For version 1, use:
- research papers
- technical deliverables
- proposal-style documents

### User input

The user will enter:
- one natural-language question

Examples:
- `What are the main contributions of the SUNRISE validation paper?`
- `What latency values were observed in the hybrid testing setup?`
- `How does the GNSS spoofing detector work?`
- `What are the key results of the CPM spoof detection model?`

## Internal Processing

The pipeline should do this:

1. read PDFs
2. extract text
3. split text into chunks
4. create embeddings
5. store chunks in a vector DB
6. retrieve top relevant chunks for a question
7. generate an answer grounded in retrieved context

## Main Output

The app should return:

1. `Answer`
   - short, clear, grounded response

2. `Citations`
   - file name
   - chunk excerpt or source section

3. `Retrieved context`
   - top relevant chunks

## Output Format For MVP

Use this simple format:

- Answer
- Sources
- Retrieved chunks

Example:

Answer:
`The paper shows strong agreement between virtual and hybrid testing, with speed deviations below 2 km/h and trigger distance differences under 3 m.`

Sources:
- `vehicles-4060223.pdf`

Retrieved chunks:
- chunk 4
- chunk 7

## Optional Output Later

After MVP, you can add:
- JSON extraction mode
- summary mode
- compare-documents mode
- keyword extraction
- table extraction

## Acceptance Test

The input/output design is acceptable if:
- a non-technical person can understand how to use the app
- a technical reviewer can verify where the answer came from
- the output is simple enough to demo in under `2` minutes
