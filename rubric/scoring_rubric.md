# EB-1A / O-1A Case Quality Scoring Rubric

## Auto-Reject (Score 0)
Assign a score of 0 and set `auto_reject: true` if ANY of the following apply:
- Untimely filed appeal (dismissed for procedural reasons)
- Motion to reopen/reconsider with no substantive legal reanalysis
- Jurisdictional dismissal (AAO lacks jurisdiction)
- Withdrawn appeal
- Substantive content is under 500 words (excluding headers, footers, redaction notices)

If auto-rejected, set `rejection_reason` to the specific reason above.

## Scoring Criteria (1-10 scale)

### 1. Criteria Coverage (30%)
How many of the 8 EB-1A/O-1A evidentiary criteria are substantively discussed?
- **1-2:** Only 1 criterion mentioned
- **3-4:** 2-3 criteria discussed
- **5-6:** 4-5 criteria discussed
- **7-8:** 6-7 criteria discussed
- **9-10:** All 8 criteria addressed

The 8 evidentiary criteria are:
1. Awards — nationally/internationally recognized prizes or awards for excellence
2. Membership — membership in associations requiring outstanding achievements
3. Published material — published material in professional publications about the person
4. Judging — participation as a judge of the work of others in the field
5. Original contributions — original scientific, scholarly, or business-related contributions of major significance
6. Scholarly articles — authorship of scholarly articles in professional journals or major media
7. Exhibition/showcase — display of work at artistic exhibitions or showcases
8. Leading/critical role — performance in a leading or critical role for organizations with a distinguished reputation
9. High salary — commanding a high salary or other remuneration relative to others in the field
10. Commercial success — commercial successes in the performing arts

Note: Only 3 of these 10 need to be met for approval. The criteria numbering above includes all regulatory criteria; some cases may reference subsets.

### 2. Analytical Depth (30%)
Does the decision explain *why* evidence met or failed each criterion?
- **1-3:** Boilerplate language, conclusory statements with no reasoning
- **4-6:** Some reasoning but surface-level; states what evidence was submitted but not why it's sufficient/insufficient
- **7-8:** Detailed analysis explaining the gap between submitted evidence and the standard
- **9-10:** Thorough, multi-paragraph analysis per criterion with specific reasoning about evidentiary weight

### 3. Evidence Discussion (20%)
Does the decision reference specific evidence?
- **1-3:** Generic references ("the petitioner submitted evidence")
- **4-6:** Names some evidence types but lacks specifics
- **7-8:** References specific documents, publications, awards, salary figures, or expert letters
- **9-10:** Detailed discussion of multiple pieces of evidence with specifics

### 4. Legal Reasoning (10%)
Does the decision cite relevant law and precedent?
- **1-3:** No legal citations beyond the form type
- **4-6:** Cites INA section or 8 CFR but no case law
- **7-8:** Cites relevant precedent (e.g., Kazarian v. USCIS, Dhanasar)
- **9-10:** Multiple legal citations with explanation of how precedent applies

### 5. Outcome Clarity (10%)
Is the decision and its rationale clear?
- **1-3:** Unclear outcome or reasoning
- **4-6:** Outcome stated but rationale is weak
- **7-8:** Clear outcome with stated reasoning
- **9-10:** Unambiguous outcome with comprehensive rationale

## Output Format

Return a JSON object with exactly these fields:
```json
{
  "filename": "the source filename",
  "score": 7.5,
  "outcome": "dismiss",  // valid values: "sustain", "dismiss", "remand", "approve"
  "criteria_discussed": ["awards", "membership", "original contributions", "scholarly articles"],
  "auto_reject": false,
  "rejection_reason": null,
  "summary": "One paragraph summarizing the case facts and decision."
}
```

For the `criteria_discussed` array, use these exact string values:
- "awards"
- "membership"
- "published material"
- "judging"
- "original contributions"
- "scholarly articles"
- "exhibition"
- "leading role"
- "high salary"
- "commercial success"
