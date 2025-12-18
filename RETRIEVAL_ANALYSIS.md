# Retrieval Analysis for Problem Questions

## Target Questions

### Question 1: Scott Derrickson / Ed Wood
**Question**: "Were Scott Derrickson and Ed Wood of the same nationality?"  
**Ground Truth**: "yes"  
**Type**: comparison

**Required Articles**:
1. `Scott Derrickson` - "Scott Derrickson (born July 16, 1966) is an **American** director..."
2. `Ed Wood` - "Edward Davis Wood Jr. ... was an **American** filmmaker..."

**Current Failure Mode**:
- Step 1: RETRIEVE_KEYWORD["Scott Derrickson nationality"] → Found Scott Derrickson article ✓
- Step 2: GENERATE_SLM → HALLUCINATED "Ed Wood was Canadian" without retrieving Ed Wood article ✗

**Problem**: Agent jumped to generate without retrieving the second entity.

---

### Question 2: Kiss and Tell / Shirley Temple
**Question**: "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?"  
**Ground Truth**: "Chief of Protocol"  
**Type**: bridge

**Required Articles**:
1. `Kiss and Tell (1945 film)` - "...starring then 17-year-old **Shirley Temple** as Corliss Archer"
2. `Shirley Temple` - "...served as **Chief of Protocol** of the United States"

**Current Failure Mode**:
- Step 1: RETRIEVE_KEYWORD["actress who played Corliss Archer..."] → Found Kiss and Tell article ✓
- Step 2: GENERATE_SLM["What government position was held by Janet Waldo?"] → Wrong person! ✗

**Problem**: 
1. Agent correctly found Shirley Temple's name in context
2. But then asked about "Janet Waldo" (confusion/hallucination)
3. Never retrieved Shirley Temple's article to find her government position

---

## Retrieval Testing Results (BM25)

### Question 1 Queries:
| Query | Top 3 Results | Found Target? |
|-------|---------------|---------------|
| "Scott Derrickson" | Sinister (film), **Scott Derrickson**, Sinister 2 | ✓ |
| "Ed Wood" | Woodson Arkansas, Ed Wood (film), Conrad Brooks | ✗ |
| "Ed Wood nationality" | Woodson Arkansas, Ed Wood (film), Conrad Brooks | ✗ |
| "Ed Wood American filmmaker" | Ed Wood (film), **Ed Wood**, Woodson Arkansas | ✓ |

**Key Finding**: BM25 fails on "Ed Wood" alone because:
- "Ed" and "Wood" are common words
- Need domain-specific terms like "filmmaker" or "director" to disambiguate

### Question 2 Queries:
(Need to test: Kiss and Tell, Corliss Archer, Shirley Temple, Shirley Temple government)

---

## Proposed Solutions

### Solution 1: DECOMPOSE before RETRIEVE for comparison questions
For "Were X and Y same Z?", decompose into:
1. "What is X's Z?" 
2. "What is Y's Z?"

Then retrieve for each sub-question separately.

### Solution 2: Entity-aware retrieval queries
When retrieving for a person, add their profession/domain:
- "Ed Wood" → "Ed Wood filmmaker" or "Ed Wood director"
- "Scott Derrickson" → "Scott Derrickson director"

### Solution 3: Two-pass retrieval
1. First pass: Retrieve with entity name
2. If no good match, retry with entity + domain terms

### Solution 4: Force second retrieval for comparison questions
After first retrieval, check if BOTH entities have info in context.
If not, force another RETRIEVE for the missing entity.
