# {PANEL_NAME} Panel

## Overview

{PANEL_DESCRIPTION}

## Input Data

| Source | Description |
|--------|-------------|
| `{INPUT_FILE}` | {DESCRIPTION} |

## Analysis Method

### Statistical Approach

{STATISTICAL_METHOD}

### Code Reference

```python
# Function: {FUNCTION_NAME}()
# File: {SCRIPT_PATH}:{LINE_NUMBER}
```

## Output Schema

### CSV Output

| Column | Type | Description |
|--------|------|-------------|
| `{COL_NAME}` | {TYPE} | {DESCRIPTION} |

### JSON Output

```json
{
  "{FIELD}": "{DESCRIPTION}"
}
```

## Visualization

### UI Component

- **Tab**: {TAB_NAME}
- **Panel**: {PANEL_NAME}
- **Update Function**: `{UPDATE_FUNCTION}()`

### API Endpoint

```
GET /api/v1/{ATLAS}/{ENDPOINT}
```

## Example Results

{EXAMPLE_RESULTS}

## Interpretation

{INTERPRETATION_GUIDE}

## Related

- [Parent Pipeline](../{ATLAS}/activity.md)
- [JSON Output](../../outputs/visualization/{JSON_FILE}.md)
