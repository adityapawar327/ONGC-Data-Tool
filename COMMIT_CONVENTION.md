# Commit Message Convention

This repository follows a standardized commit message format to maintain a clear and meaningful git history.

## Commit Message Format
```
<type>(<scope>): <short summary>
<BLANK LINE>
<body>
<BLANK LINE>
<footer>
```

## Type

Must be one of the following:

* **feat**: ✨ A new feature
* **fix**: 🐛 A bug fix
* **docs**: 📚 Documentation only changes
* **style**: 💎 Changes that do not affect the meaning of the code (formatting, etc)
* **refactor**: 📦 A code change that neither fixes a bug nor adds a feature
* **perf**: 🚀 A code change that improves performance
* **test**: 🧪 Adding missing tests or correcting existing tests
* **chore**: 🔧 Changes to build process or auxiliary tools
* **revert**: ⏪️ Revert to a previous commit
* **data**: 📊 Changes related to data processing or analysis
* **ai**: 🤖 Changes to AI/ML models or algorithms
* **clean**: 🧹 Data cleaning and preprocessing changes

## Scope (Optional)

Common scopes for this project:
* **analysis**: Data analysis functionality
* **clean**: Data cleaning operations
* **convert**: File conversion features
* **search**: Search functionality
* **ai**: AI/ML components
* **label**: Data labeling
* **compare**: Comparison features

## Summary

* Use imperative, present tense: "change" not "changed" nor "changes"
* Don't capitalize the first letter
* No period (.) at the end

## Body (Optional)

* Use imperative, present tense
* Include motivation for the change
* Contrast this with previous behavior

## Footer (Optional)

* Reference issues the commit closes
* Examples:
  * `Closes #123`
  * `Fixes #123`
  * `Related to #123`

## Examples

```
feat(analysis): add new data visualization component

Implement interactive charts for well data analysis.
This allows users to visualize depth vs. pressure relationships.

Closes #45
```

```
fix(clean): correct date format parsing in well logs

Update the data cleaning logic to handle different date formats
from various well log sources properly.

Fixes #67
```

```
data(analysis): update normalization parameters

Adjust the normalization factors for pressure data
to better account for depth variations.
```

```
ai(model): improve well prediction accuracy

Fine-tune the machine learning model parameters
to reduce prediction error by 15%.
```

## Using the Commit Template

A commit template has been set up. When committing, you'll see the template in your editor.
Fill in the relevant sections and remove the comment lines (starting with #).
