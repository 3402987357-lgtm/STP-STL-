$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path.TrimEnd("\")
$cacheNames = @("__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache")
$cacheDirectories = Get-ChildItem -LiteralPath $root -Recurse -Directory -Force |
    Where-Object { $_.Name -in $cacheNames }

foreach ($directory in $cacheDirectories) {
    $fullPath = $directory.FullName.TrimEnd("\")
    if (
        $fullPath -eq $root -or
        -not $fullPath.StartsWith($root + "\", [StringComparison]::OrdinalIgnoreCase)
    ) {
        throw "Refusing to delete a cache outside the project root: $fullPath"
    }
    Remove-Item -LiteralPath $fullPath -Recurse -Force
}

$remaining = Get-ChildItem -LiteralPath $root -Recurse -Directory -Force |
    Where-Object { $_.Name -in $cacheNames }
if ($remaining) {
    $remaining | Select-Object -ExpandProperty FullName
    throw "Cache directories remain after cleanup."
}

Write-Output "Project cache directories: 0"
