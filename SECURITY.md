# Security Policy

## Reporting a Vulnerability

Please report suspected vulnerabilities privately through GitHub's private
vulnerability reporting feature. Do not include credentials, private datasets,
or exploitable details in a public issue.

## Credential Handling

SAG reads API credentials from environment variables or command-line options.
Never commit real credentials, `.env` files, private keys, generated client or
server configurations, logs, or private knowledge-base material.

Any credential that has appeared in source code or Git history must be revoked
and replaced; deleting it from the latest revision is not sufficient.

## Security Scope

The formal guarantees and deployment assumptions are defined in the associated
papers. This research implementation has not undergone a production security
audit and should not be used as the sole protection for sensitive production
data.
