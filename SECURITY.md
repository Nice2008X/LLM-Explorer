# Security Policy

## Reporting a Vulnerability

Please **do not** report security vulnerabilities through public GitHub issues, discussions, or pull requests.

Instead, report them privately using one of the following channels:

1. **GitHub Private Vulnerability Reporting (preferred)**
   Go to the [Security tab](https://github.com/Nice2008X/Tensorium/security) of this repository and select **"Report a vulnerability"** to open a private advisory. This keeps the report and any discussion confidential until a fix is available.

2. **Email**
   If you're unable to use GitHub's private reporting, email **aixu2008nice@gmail.com** with details of the issue. Please include "SECURITY" in the subject line.

Please include as much of the following as you can:

- A description of the vulnerability and its potential impact
- Steps to reproduce, including a minimal example if possible
- The affected version/commit
- Any suggested fix or mitigation, if known

### What to expect

- Acknowledgement of your report within a few days.
- An initial assessment of the issue and its severity.
- Coordination on a fix and disclosure timeline before any public details are shared. We ask that you give us reasonable time to address the issue before any public disclosure.

### Scope

Tensorium is a client-side, browser-only application — model files are loaded and processed entirely in the user's browser, with no backend server or user data storage. Reports of particular interest include (but aren't limited to):

- Cross-site scripting (XSS) or other injection issues in the UI
- Unsafe parsing of untrusted model/config/weight files (e.g. `config.json`, `.safetensors`) that could lead to code execution or memory corruption
- Dependency vulnerabilities with a credible exploit path in this project
- Supply-chain issues in the build or release process

Thank you for helping keep Tensorium and its users safe.
