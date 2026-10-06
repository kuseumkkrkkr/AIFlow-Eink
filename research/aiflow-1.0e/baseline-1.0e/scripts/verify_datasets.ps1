param(
  [string]$DatasetRoot = (Join-Path $PSScriptRoot '..\datasets'),
  [switch]$Full
)

$ErrorActionPreference = 'Stop'
$resolvedDatasetRoot = [IO.Path]::GetFullPath($DatasetRoot)
if (-not $resolvedDatasetRoot.StartsWith('D:\', [StringComparison]::OrdinalIgnoreCase)) {
  throw "Dataset verification must run against D: ($resolvedDatasetRoot)"
}

$expected = [ordered]@{
  '10_approved_external\uji_pen_characters_v2\raw\uji_pen_v2.zip' = '0881B522911B99D9922820289441B50FD3D307F71CD7F9CC70E86872424A5F90'
  '10_approved_external\uji_pen_characters_v2\derived\uji_math_curated.jsonl.gz' = 'CDA2FE17C213BC8C90ED93EC8998658DDAF9D7871C75ADF521798E6E3BA3029B'
  '10_approved_external\isgl_online_offline_hwr\raw\isgl_source.zip' = 'A94F2473246222F9470D4B93B68CFBC756ECB4865742DB8164788359FE511693'
  '10_approved_external\isgl_online_offline_hwr\migrated\isgl_online.jsonl.gz' = '0A4BD4FE37CD7656B2FD3586FED4D5C886FEBA1D8E5120803B60A0F04044988D'
  '10_approved_external\uci_character_trajectories\raw\character_trajectories.zip' = '5D2DB017EF0D8CF0E65ED060C9E90399F78EB9F1E3CB63E22CA8C3EF4BA67D52'
  '10_approved_external\uci_character_trajectories\raw\character+trajectories.official.zip' = '5D2DB017EF0D8CF0E65ED060C9E90399F78EB9F1E3CB63E22CA8C3EF4BA67D52'
  '10_approved_external\hwrt\raw\2015-01-28-data.tar' = 'B96FEAFD71B01F1623997DFF3CC8AC4D18628D128CEE1B3DF1880518BBA3EA4A'
  '10_approved_external\hwrt\derived\train.jsonl.gz' = '4C067A06FFAB8A73FE99C10B633B1175A68E512D17CA68D587F56E6182AD2561'
  '10_approved_external\hwrt\derived\validation.jsonl.gz' = '9813260509A4025E203CD36A3AE08C8A6CBC342233684C12C13B2C447D8B0A34'
  '10_approved_external\hwrt\derived\test.jsonl.gz' = '902EAAA1311708D4A73F1E63A9C88AE15CEDDED945E7B915B7CE177DAD30AF8A'
  '10_approved_external\hwrt\derived\rejections.jsonl.gz' = '47B0968D8387E10DED1413F74D46B504FA560F4F6CBD4CBE81ED7D859AFB6644'
  '10_approved_external\bdshwa\raw\bdshwa_v1.zip' = '45FBBDCD1A9C4353F5C93BF8096FBEE9723931185F7BBFED3A86E21BA4CF20EA'
  '20_reaudit_required\hf_cli_unlicensed\handwriting_strokes\raw\data\train-00000-of-00001.parquet' = 'E8E11649A4F8E1262ED0B474D301D7761ACFD8BB015A44FE2C6D573649822CEC'
  '20_reaudit_required\hf_cli_unlicensed\handwriting_strokes\raw\data\validation-00000-of-00001.parquet' = 'A5E4AC5FA1C8327284B2D927A1CFC32391E1F6297A989208974D5B24AE68CAED'
  '20_reaudit_required\hf_cli_unlicensed\handwriting_strokes\raw\data\test-00000-of-00001.parquet' = '21297408C969D5985CF96F00610F735E82D1C5A0E01F1EEF62B2F3576B37E972'
  '20_reaudit_required\hf_cli_unlicensed\handwriting_strokes_2strokes\raw\data\train-00000-of-00001.parquet' = 'A90988B17975FEDAE5C0E26943058134D3F8D20F09BD399EF0ADFB0C728F8512'
  '20_reaudit_required\hf_cli_unlicensed\handwriting_strokes_2strokes\raw\data\validation-00000-of-00001.parquet' = 'C83EED7E1D3952CF6419C149F35E3961A7ED059DED97A60724E071013286DE14'
  '20_reaudit_required\hf_cli_unlicensed\handwriting_strokes_2strokes\raw\data\test-00000-of-00001.parquet' = 'C83EED7E1D3952CF6419C149F35E3961A7ED059DED97A60724E071013286DE14'
  '20_reaudit_required\hf_cli_unlicensed\edge_case_1stroke\raw\data\train-00000-of-00001.parquet' = 'A331C2A3F06C3EB7F85191E3C1079AD4059D43447CB6E22B1C5A04397817AC15'
  '20_reaudit_required\hf_cli_unlicensed\edge_case_1stroke\raw\data\validation-00000-of-00001.parquet' = 'EE742E79F70E2E644D37185F5B45F4EE4D740016277C28BBFE9D8DD88CC3F1E0'
  '20_reaudit_required\hf_cli_unlicensed\edge_case_1stroke\raw\data\test-00000-of-00001.parquet' = '2208696803A05A0B7DE5CEC37729521FD0A25240BCC6E90401D8864ABC3DC85E'
}

$largeArtifactLengths = @{
  '10_approved_external\isgl_online_offline_hwr\raw\isgl_source.zip' = [int64]974053582
  '10_approved_external\hwrt\raw\2015-01-28-data.tar' = [int64]140790596
  '10_approved_external\hwrt\derived\train.jsonl.gz' = [int64]153885264
  '10_approved_external\bdshwa\raw\bdshwa_v1.zip' = [int64]1318392171
}

$hashedCount = 0
$sizeOnlyCount = 0
foreach ($entry in $expected.GetEnumerator()) {
  $path = Join-Path $resolvedDatasetRoot $entry.Key
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing dataset artifact: $path" }
  if (-not $Full -and $largeArtifactLengths.ContainsKey($entry.Key)) {
    if ((Get-Item -LiteralPath $path).Length -ne $largeArtifactLengths[$entry.Key]) { throw "Length mismatch: $path" }
    $sizeOnlyCount++
  }
  else {
    if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $entry.Value) { throw "Hash mismatch: $path" }
    $hashedCount++
  }
}

Add-Type -AssemblyName System.IO.Compression.FileSystem
$zipFiles = @(
  '10_approved_external\uji_pen_characters_v2\raw\uji_pen_v2.zip',
  '10_approved_external\isgl_online_offline_hwr\raw\isgl_source.zip',
  '10_approved_external\uci_character_trajectories\raw\character_trajectories.zip',
  '10_approved_external\uci_character_trajectories\raw\character+trajectories.official.zip',
  '10_approved_external\bdshwa\raw\bdshwa_v1.zip'
)
foreach ($relative in $zipFiles) {
  $archive = [IO.Compression.ZipFile]::OpenRead((Join-Path $resolvedDatasetRoot $relative))
  try {
    if ($archive.Entries.Count -lt 1) { throw "Empty ZIP: $relative" }
  }
  finally { $archive.Dispose() }
}

$buffer = New-Object byte[] 1048576
foreach ($relative in @(
  '10_approved_external\uji_pen_characters_v2\derived\uji_math_curated.jsonl.gz',
  '10_approved_external\isgl_online_offline_hwr\migrated\isgl_online.jsonl.gz',
  '10_approved_external\hwrt\derived\validation.jsonl.gz',
  '10_approved_external\hwrt\derived\test.jsonl.gz',
  '10_approved_external\hwrt\derived\rejections.jsonl.gz'
)) {
  $input = [IO.File]::OpenRead((Join-Path $resolvedDatasetRoot $relative))
  $gzip = [IO.Compression.GZipStream]::new($input, [IO.Compression.CompressionMode]::Decompress)
  try { while ($gzip.Read($buffer, 0, $buffer.Length) -gt 0) {} }
  finally {
    $gzip.Dispose()
    $input.Dispose()
  }
}

$hwrtTrain = Join-Path $resolvedDatasetRoot '10_approved_external\hwrt\derived\train.jsonl.gz'
$input = [IO.File]::OpenRead($hwrtTrain)
$gzip = [IO.Compression.GZipStream]::new($input, [IO.Compression.CompressionMode]::Decompress)
try {
  if ($gzip.Read($buffer, 0, $buffer.Length) -lt 1) { throw "Empty HWRT train GZIP: $hwrtTrain" }
}
finally {
  $gzip.Dispose()
  $input.Dispose()
}

$manifestPath = Join-Path $resolvedDatasetRoot '10_approved_external\hwrt\derived\manifest.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($manifest.counts.accepted_total -ne 168027 -or $manifest.counts.rejected_total -ne 206) {
  throw "Unexpected HWRT manifest row counts"
}
if ($manifest.counts.rejection_reasons.timestamp_reversal -ne 193 -or $manifest.counts.rejection_reasons.duplicate_sample -ne 13) {
  throw "Unexpected HWRT rejection counts"
}
if ($manifest.counts.contributor_group_types.detexify_aggregate_unknown_writers -ne 153660) {
  throw "Unexpected HWRT Detexify aggregate count"
}
if ($manifest.counts.output_timestamp_reversals -ne 0) { throw "HWRT output timestamp reversal recorded" }
$overlap = $manifest.counts.source_group_overlap
if ($overlap.train_validation -ne 0 -or $overlap.train_test -ne 0 -or $overlap.validation_test -ne 0) {
  throw "HWRT source-group split overlap recorded"
}
if ($manifest.source_user_id_caveat.true_writer_disjoint -ne 'unverifiable') {
  throw "HWRT true-writer caveat missing"
}

$tarPath = Join-Path $resolvedDatasetRoot '10_approved_external\hwrt\raw\2015-01-28-data.tar'
$tarEntries = @(& tar -tf $tarPath)
if ($LASTEXITCODE -ne 0) { throw "Unreadable TAR: $tarPath" }
$sortedTarEntries = @($tarEntries | Sort-Object)
if (($sortedTarEntries -join ',') -ne 'symbols.csv,test-data.csv,train-data.csv') {
  throw "Unexpected HWRT TAR members: $($tarEntries -join ', ')"
}

foreach ($relative in @($expected.Keys | Where-Object { $_ -like '*.parquet' })) {
  $path = Join-Path $resolvedDatasetRoot $relative
  $stream = [IO.File]::OpenRead($path)
  try {
    $magic = New-Object byte[] 4
    if ($stream.Read($magic, 0, 4) -ne 4 -or [Text.Encoding]::ASCII.GetString($magic) -ne 'PAR1') {
      throw "Invalid Parquet header: $relative"
    }
    $stream.Seek(-4, [IO.SeekOrigin]::End) | Out-Null
    if ($stream.Read($magic, 0, 4) -ne 4 -or [Text.Encoding]::ASCII.GetString($magic) -ne 'PAR1') {
      throw "Invalid Parquet footer: $relative"
    }
  }
  finally { $stream.Dispose() }
}

if ($Full) {
  & python (Join-Path $PSScriptRoot 'build_hwrt_curated.py') --verify-only
  if ($LASTEXITCODE -ne 0) { throw "Full HWRT derivative validation failed" }
}

$mode = if ($Full) { 'full' } else { 'quick' }
Write-Output "Verified $($expected.Count) D: artifacts ($hashedCount hashes, $sizeOnlyCount audited large-file lengths), five ZIP catalogs, six GZIP containers, one TAR, one HWRT manifest, and nine Parquet files in $mode mode."
