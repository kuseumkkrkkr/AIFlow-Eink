[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$SourceUrl,

    [Parameter(Mandatory = $true)]
    [string]$OutputPath,

    [Parameter(Mandatory = $true)]
    [long]$ExpectedLength,

    [ValidateRange(1, 32)]
    [int]$SegmentCount = 16,

    [ValidateRange(1, 32)]
    [int]$ParallelMax = 8
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$resolvedOutput = [System.IO.Path]::GetFullPath($OutputPath)
if (-not $resolvedOutput.StartsWith('D:\', [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Dataset output must stay on D: ($resolvedOutput)"
}

$parent = [System.IO.Path]::GetDirectoryName($resolvedOutput)
[System.IO.Directory]::CreateDirectory($parent) | Out-Null

$partsDirectory = "$resolvedOutput.parts"
[System.IO.Directory]::CreateDirectory($partsDirectory) | Out-Null

$segmentSize = [long][Math]::Ceiling($ExpectedLength / [double]$SegmentCount)
$segments = for ($index = 0; $index -lt $SegmentCount; $index++) {
    $start = [long]$index * $segmentSize
    if ($start -ge $ExpectedLength) { break }
    $end = [Math]::Min($ExpectedLength - 1, $start + $segmentSize - 1)
    $partPath = Join-Path $partsDirectory ('part-{0:D3}-{1}-{2}.bin' -f $index, $start, $end)
    [pscustomobject]@{
        Index = $index
        Start = $start
        End = $end
        Length = $end - $start + 1
        Path = $partPath
    }
}

$pending = @()
foreach ($segment in $segments) {
    if (Test-Path -LiteralPath $segment.Path) {
        $actualLength = (Get-Item -LiteralPath $segment.Path).Length
        if ($actualLength -eq $segment.Length) { continue }
        $invalidPath = "$($segment.Path).invalid-$([DateTime]::UtcNow.ToString('yyyyMMddHHmmssfff'))"
        Move-Item -LiteralPath $segment.Path -Destination $invalidPath
    }
    $pending += $segment
}

if ($pending.Count -gt 0) {
    $curlArguments = @('--parallel', '--parallel-immediate', '--parallel-max', "$ParallelMax")
    for ($pendingIndex = 0; $pendingIndex -lt $pending.Count; $pendingIndex++) {
        $segment = $pending[$pendingIndex]
        $curlArguments += @(
            '--fail',
            '--location',
            '--retry', '8',
            '--retry-delay', '3',
            '--retry-all-errors',
            '--silent',
            '--show-error',
            '--range', "$($segment.Start)-$($segment.End)",
            '--output', $segment.Path,
            $SourceUrl
        )
        if ($pendingIndex -lt $pending.Count - 1) {
            $curlArguments += '--next'
        }
    }

    & curl.exe @curlArguments
    if ($LASTEXITCODE -ne 0) {
        throw "curl range download failed with exit code $LASTEXITCODE"
    }
}

foreach ($segment in $segments) {
    if (-not (Test-Path -LiteralPath $segment.Path)) {
        throw "Missing range part: $($segment.Path)"
    }
    $actualLength = (Get-Item -LiteralPath $segment.Path).Length
    if ($actualLength -ne $segment.Length) {
        throw "Range part length mismatch: $($segment.Path) expected=$($segment.Length) actual=$actualLength"
    }
}

$assemblingPath = "$resolvedOutput.assembling"
if (Test-Path -LiteralPath $assemblingPath) {
    $invalidAssembly = "$assemblingPath.invalid-$([DateTime]::UtcNow.ToString('yyyyMMddHHmmssfff'))"
    Move-Item -LiteralPath $assemblingPath -Destination $invalidAssembly
}

$outputStream = [System.IO.File]::Create($assemblingPath)
try {
    foreach ($segment in ($segments | Sort-Object Index)) {
        $inputStream = [System.IO.File]::OpenRead($segment.Path)
        try {
            $inputStream.CopyTo($outputStream)
        }
        finally {
            $inputStream.Dispose()
        }
    }
}
finally {
    $outputStream.Dispose()
}

$assembledLength = (Get-Item -LiteralPath $assemblingPath).Length
if ($assembledLength -ne $ExpectedLength) {
    throw "Assembled file length mismatch: expected=$ExpectedLength actual=$assembledLength"
}

if (Test-Path -LiteralPath $resolvedOutput) {
    $existingBackup = "$resolvedOutput.pre-range-$([DateTime]::UtcNow.ToString('yyyyMMddHHmmssfff')).partial"
    Move-Item -LiteralPath $resolvedOutput -Destination $existingBackup
}
Move-Item -LiteralPath $assemblingPath -Destination $resolvedOutput

$hash = Get-FileHash -LiteralPath $resolvedOutput -Algorithm SHA256
[pscustomobject]@{
    Path = $resolvedOutput
    Length = $assembledLength
    SHA256 = $hash.Hash
    Segments = $segments.Count
}
