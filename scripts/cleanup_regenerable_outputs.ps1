param(
    # 我先看清"要删什么、能省多少"，再决定真删。
    # 清理是不可逆的，预演不花代价。
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path.TrimEnd("\")
$outputRoot = Join-Path $root "outputs"
$targets = @(
    "extreme_probe",
    "extreme_round3",
    "extreme_round4_parting",
    "extreme_round4_probe",
    "final_validation",
    "real_pair_stress_2026-10-02",
    "real_step_validation",
    "step_final",
    "parting_bottle_candidate_v2",
    "parting_box_hardened",
    "parting_cli_review_probe",
    "parting_demo",
    "parting_baseline",
    "smoke_step_repair_bottle",
    "smoke_step_repair_mould04",
    "smoke_step_continuity_bottle",
    "smoke_step_continuity_mould04",
    "inspect_demo",
    "demo",
    "final_check",
    "probe",
    "probe_cli",
    "probe_final",
    "probe_real",
    "_smoke_region_signature_20261003",
    "_calib_probe",
    "_bom_split_test",
    # 我核对四个区域 CLI 端到端跑通时造的探针产物，可随时重跑。
    "_probe\region_cli",
    "outputs_audit.json",
    "hardening_outputs_audit_20261003.json",
    "hardening_outputs_audit_20261004.json",
    "hardening_cli_20261003",
    "hardening_stress_20261003",
    "hardening_stress_ocp_20261003",
    "features",
    "repair_demo",
    "step_engineering\BOTTLE.engineering.json",
    "step_engineering\BOTTLE.engineering.csv",
    "step_engineering\BOTTLE.regions.csv",
    "step_engineering\BOTTLE.draft.csv",
    "step_engineering\BOTTLE\_regeneration.log",
    "step_engineering\mould_04.engineering.json",
    "step_engineering\mould_04.engineering.csv",
    "step_engineering\mould_04.regions.csv",
    "step_engineering\mould_04.draft.csv",
    "step_engineering\mould_04\_regeneration.log",
    "step_engineering\_regeneration.log",
    "step_engineering\_regeneration_mould_04.log",
    "stage3_regions_v3\BOTTLE_regeneration.log",
    "stage3_regions_v3\mould_04_regeneration.log"
)

# 我再按通配符兜底：任何 "*regeneration*.log" 都只是可再生成的过程日志，
# 不属于正式交付物，统一收口，避免后续运行把 D 盘堆满。
$regenerationLogs = Get-ChildItem -LiteralPath $outputRoot -Recurse -File -Force |
    Where-Object { $_.Name -like "*regeneration*.log" }

# STEP 缺面基准的中间几何（damaged.stp / damaged.repaired.step）体积大且
# 随时可重跑，我按通配符收口；根目录的 summary.json 与 summary.csv 是正式
# 证据，必须留下，所以只匹配子目录里的中间文件。
$benchmarkIntermediates = Get-ChildItem -LiteralPath $outputRoot -Recurse -File -Force |
    Where-Object {
        $_.FullName -like "*step_repair_benchmark*\*" -and
        $_.Name -like "damaged*"
    }

$formalTargets = @(
    "bom_bottle\BOTTLE.bom.json",
    "bom_bottle\BOTTLE.bom.csv",
    "bom_bottle\BOTTLE.bom.xlsx",
    # 这三个是 regression 测试**当输入读**的基准产物，不是过程产物。
    # 它们一旦被下面的 damaged* 通配规则删掉，对应测试会走 pytest.skip，
    # 套件仍然全绿、覆盖率却静默掉了。所以钉进保护名单，
    # 并由 tests/test_cleanup_policy.py 反向看住。
    "step_repair_benchmark_relaxed\测试集__10_face3\damaged.stp",
    "step_repair_benchmark_relaxed\测试集__11_face3\damaged.repaired.step",
    "step_engineering\BOTTLE\BOTTLE.engineering.json",
    "step_engineering\BOTTLE\BOTTLE.engineering.csv",
    "step_engineering\BOTTLE\BOTTLE.regions.csv",
    "step_engineering\BOTTLE\BOTTLE.draft.csv",
    "step_engineering\mould_04\mould_04.engineering.json",
    "step_engineering\mould_04\mould_04.engineering.csv",
    "step_engineering\mould_04\mould_04.regions.csv",
    "step_engineering\mould_04\mould_04.draft.csv"
)
$protectedPaths = @()
foreach ($relativeTarget in $formalTargets) {
    $target = Join-Path $outputRoot $relativeTarget
    if (-not (Test-Path -LiteralPath $target -PathType Leaf)) {
        throw "我方拒绝清理：正式工程输出不存在：$target"
    }
    $protectedPaths += (Resolve-Path -LiteralPath $target).Path.TrimEnd("\")
}

$resolvedTargets = @()
foreach ($relativeTarget in $targets) {
    $target = Join-Path $outputRoot $relativeTarget
    $item = Get-Item -LiteralPath $target -Force -ErrorAction SilentlyContinue
    if ($null -eq $item) {
        continue
    }
    $fullPath = $item.FullName.TrimEnd("\")
    if (
        $fullPath -eq $root -or
        -not $fullPath.StartsWith($outputRoot.TrimEnd("\") + "\", [StringComparison]::OrdinalIgnoreCase)
    ) {
        throw "Refusing to delete a target outside the outputs directory: $fullPath"
    }
    $resolvedTargets += $item
}

foreach ($item in $regenerationLogs) {
    $fullPath = $item.FullName.TrimEnd("\")
    if ($protectedPaths -contains $fullPath) {
        continue
    }
    if (
        $fullPath -eq $root -or
        -not $fullPath.StartsWith($outputRoot.TrimEnd("\") + "\", [StringComparison]::OrdinalIgnoreCase)
    ) {
        throw "Refusing to delete a target outside the outputs directory: $fullPath"
    }
    if (-not ($resolvedTargets | Where-Object { $_.FullName -eq $fullPath })) {
        $resolvedTargets += $item
    }
}

foreach ($item in $benchmarkIntermediates) {
    $fullPath = $item.FullName.TrimEnd("\")
    # 被 $formalTargets 保护的文件也可能落进 damaged* 通配规则，
    # 我显式跳过，否则"保护名单"对通配规则无效。
    if ($protectedPaths -contains $fullPath) {
        continue
    }
    if (
        $fullPath -eq $root -or
        -not $fullPath.StartsWith($outputRoot.TrimEnd("\") + "\", [StringComparison]::OrdinalIgnoreCase)
    ) {
        throw "Refusing to delete a target outside the outputs directory: $fullPath"
    }
    if (-not ($resolvedTargets | Where-Object { $_.FullName -eq $fullPath })) {
        $resolvedTargets += $item
    }
}

$reportLines = $resolvedTargets | ForEach-Object {
    if ($_.PSIsContainer) {
        $children = Get-ChildItem -LiteralPath $_.FullName -Recurse -File -Force
        "{0,10:N1} MB  {1}" -f (
            ($children | Measure-Object -Property Length -Sum).Sum / 1MB
        ), $_.FullName.Replace("$root\", "")
    } else {
        "{0,10:N1} MB  {1}" -f ($_.Length / 1MB), $_.FullName.Replace("$root\", "")
    }
}

$beforeBytes = (
    $resolvedTargets |
        ForEach-Object {
            if ($_.PSIsContainer) {
                Get-ChildItem -LiteralPath $_.FullName -Recurse -File -Force
            } else {
                $_
            }
        } |
        Measure-Object -Property Length -Sum
).Sum

if ($DryRun) {
    Write-Output "DryRun: 本次共 $($resolvedTargets.Count) 个清理目标"
    Write-Output "DryRun: 预计释放 $([math]::Round($beforeBytes / 1MB, 1)) MB"
    $reportLines | Sort-Object -Descending | Select-Object -First 30
    Write-Output "DryRun: 未删除任何文件。确认无误后去掉 -DryRun 再执行。"
    exit 0
}

foreach ($item in $resolvedTargets) {
    # 最后一道闸：任何时刻都不许删保护名单里的文件，
    # 哪怕前面的筛选逻辑将来被改坏。
    if ($protectedPaths -contains $item.FullName.TrimEnd("\")) {
        throw "我方拒绝清理受保护的正式/测试输入产物：$($item.FullName)"
    }
    Remove-Item -LiteralPath $item.FullName -Recurse -Force
}

$afterBytes = (
    Get-ChildItem -LiteralPath $outputRoot -Recurse -File -Force |
        Measure-Object -Property Length -Sum
).Sum

Write-Output "Cleanup targets: $($resolvedTargets.Count)"
Write-Output "Bytes released: $beforeBytes"
Write-Output "Outputs bytes after cleanup: $afterBytes"
