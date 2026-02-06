# Welcome to WYDOT Assistant! 

This is your AI-powered assistant for the Wyoming Department of Transportation. It uses advanced Graph based RAG (Retrieval-Augmented Generation) to answer questions based on official specifications, manuals, and reports.

##  Key Features
-  Multi-Document Search: Instantly searches across thousands of pages of WYDOT documents.
-  Multi-Hop Reasoning: Can connect facts from different sections (e.g., relating material specs to construction methods).
-  Voice Support: Click the microphone icon to ask questions verbally.
-  Multimodal Analysis: Upload images or PDFs, and the AI will analyze them using Google Gemini models.
-  Citation Tracking: Every answer provides clickable sources so you can verify the information.

## 💡 How to Ask Better Questions
 To get the **excel best** results, try to be specific and use the metadata fields our system tracks:

### 1. Filter by Year
If you need information from a specific version of the specs, mention the year.
> "What is the asphalt binder requirement in the **2021** specs?"

### 2. Reference Specific Sections
The system indexes documents by section. Referring to them helps precise retrieval.
> "Summarize **Section 101.03** regarding definitions."
> "What does **Section 401** say about mixing temperatures?"

### 3. Ask for Comparisons** 
> "Compare the aggregate gradation requirements for **Type I** vs **Type II** pavement."

### 4. Use Multimodal Capabilities** 
Upload a photo of a construction defect or a screenshot of a table and ask:
> "What does this table specify for compressive strength?"
> "Does this crack pattern indicate a subgrade failure?"

## Metadata Fields
The system tracks the following metadata for every document chunk. You can use these terms in your query to narrow down results:
- **SOURCE**: The filename of the document (e.g., `2021_Standard_Specs.pdf`)
- **TITLE**: The title of the document (e.g., `2021 Standard Specifications for Road and Bridge Construction,annual report `)
- **DOCUMENT_TYPE**: The type of document (e.g., `Standard Specifications`, `Manual`, `Report`)
- **PREVIEW**: The preview of the document (e.g., `2021 Standard Specifications for Road and Bridge Construction`)  
- **SECTION**: The section number (e.g., `401.4`)
- **YEAR**: The document publication year (e.g., `2021`, `2010`)
- **PAGE**: The page number in the original PDF


Ready to start? Type your question in the chat composer below!
