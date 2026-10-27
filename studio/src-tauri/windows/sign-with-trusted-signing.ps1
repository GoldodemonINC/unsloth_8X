# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
param(
    [Parameter(Mandatory = $true)]
    [string] $Path
)

$ErrorActionPreference = "Continue"

# Local builds can explicitly skip Azure Authenticode: the account and certificate
# belong to the hosted release environment. This does not affect the Tauri updater
# signature, which is still generated from the local updater key. Windows will show an
# unknown publisher for this local package. Do not silently skip this in CI or in a
# normal build; those must have the signing tool and credentials configured.
if ($env:UNSLOTH_SKIP_AUTHENTICODE -eq '1') {
    Write-Warning "UNSLOTH_SKIP_AUTHENTICODE=1; skipping Azure Authenticode (Tauri updater signature is still generated)."
    exit 0
}

$signer = Get-Command trusted-signing-cli -CommandType Application -ErrorAction SilentlyContinue
if (-not $signer) {
    Write-Error "trusted-signing-cli is not on PATH. Configure Azure Trusted Signing, or set UNSLOTH_SKIP_AUTHENTICODE=1 for an unsigned-publisher local build."
    exit 1
}

$maxAttempts = 3
$retryPatterns = @(
    "No subscriptions found",
    "login via azure cli",
    "az\.cmd.*exited with code 1"
)
$trustedSigningArgs = @(
    "-e",
    "https://eus.codesigning.azure.net",
    "-d",
    "Unsloth",
    $Path
)

for ($attempt = 1; $attempt -le $maxAttempts; $attempt++) {
    $output = & $signer.Path @trustedSigningArgs 2>&1
    $exitCode = $LASTEXITCODE
    if ($null -eq $exitCode) {
        $exitCode = 1
    }
    $text = $output | Out-String

    foreach ($line in $output) {
        Write-Output $line
    }

    if ($exitCode -eq 0) {
        exit 0
    }

    $isRetryable = $false
    foreach ($pattern in $retryPatterns) {
        if ($text -match $pattern) {
            $isRetryable = $true
            break
        }
    }

    if (-not $isRetryable -or $attempt -eq $maxAttempts) {
        exit $exitCode
    }

    Write-Warning "trusted-signing-cli failed with transient Azure auth error; retrying."
    Start-Sleep -Seconds (5 * $attempt)
}

exit 1
