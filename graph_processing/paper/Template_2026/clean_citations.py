import re

missing_citations = {
    'allan1998topicdetection',
    'graphragcompliance2024',
    'soman2024kgrag',
    'hybridrag2024',
    'edge2024graphrag',
    'marag2025',
    'zohar2024multimetarag',
    'routerag2025',
    'ragrouter2025',
    'chen2024blendedrag',
    'rag4cm2025',
    'barnett2024observations',
    'kwiatkowski2019naturalquestions',
    'huang2023hallucination'
}

with open("final.tex", "r") as f:
    content = f.text()

def clean_cite(match):
    citations = [c.strip() for c in match.group(1).split(',')]
    valid = [c for c in citations if c and c not in missing_citations]
    if valid:
        return f"\\cite{{{','.join(valid)}}}"
    else:
        return ""

new_content = re.sub(r'\\cite\{([^}]+)\}', clean_cite, content)

with open("final.tex", "w") as f:
    f.write(new_content)
