# Oracle Testing Suite

This directory contains test scripts for validating the 7-tier exploration Oracle system.

## Test Files

### Core Oracle Tests
- **`test_oracle_debug.py`** - Mock component testing to validate Oracle tier progression logic
- **`test_oracle_real.py`** - Real component integration using actual HotPotQA data
- **`test_oracle_real_improved.py`** - Enhanced real testing with better answer matching
- **`test_realistic_escalation.py`** - Realistic tier escalation patterns with detailed debug logs

### Legacy/Experimental
- **`test_7_tier_oracle.py`** - Early 7-tier Oracle implementation test

## Results

Test outputs are saved to `../results/testing/` with timestamped filenames.

## Usage

Run tests from the project root directory:

```bash
# Basic Oracle validation
python tests/test_oracle_debug.py

# Real component testing  
python tests/test_oracle_real_improved.py

# Detailed trajectory analysis
python tests/test_realistic_escalation.py
```

## Key Features Tested

1. **Tier Escalation** - Questions fail at insufficient tiers and succeed at appropriate complexity levels
2. **Cost Optimization** - Oracle selects cheapest successful strategy  
3. **Answer Quality** - Realistic answer generation and validation
4. **Trajectory Capture** - Complete step-by-step reasoning paths for training
5. **Document Retrieval** - Realistic document retrieval simulation across all tiers

## Test Results Format

Results include:
- Complete execution trajectories
- Cost analysis comparing alternatives  
- Retrieved document information
- Tier escalation patterns
- Training-ready data structures