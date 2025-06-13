# Commit Message Convention

This repository follows a standardized commit message format to maintain a clear and meaningful git history.

## Commit Message Format
```
<type>: <short summary>
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
feat: add user authentication system

Implement JWT-based authentication to secure API endpoints.
This allows users to log in and access protected resources.

Closes #45
```

```
fix: resolve data loading issue in dashboard

Update the data fetching logic to handle empty responses
properly and show appropriate error messages.

Fixes #67
```

```
docs: update installation instructions

Add detailed steps for setting up the development environment
and troubleshooting common issues.
```

## Using the Commit Template

A commit template has been set up. When committing, you'll see the template in your editor.
Fill in the relevant sections and remove the comment lines (starting with #).
