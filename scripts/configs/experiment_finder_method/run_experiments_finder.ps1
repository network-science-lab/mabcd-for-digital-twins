# Define the order of execution
$phases = @("find", "generate", "evaluate")

foreach ($phase in $phases) {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "PHASE: $phase" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan

    # Find files that match the current phase pattern (e.g., *find*.yaml)
    $files = Get-ChildItem -Filter "*_freebase_${phase}_*.yaml" | Sort-Object Name

    foreach ($file in $files) {
        Write-Host "Running config: $($file.Name)" -ForegroundColor Yellow
        
        # Run the command
        # We use $file.FullName to ensure the path is correct regardless of where script is run
        uv run mfdt $file.FullName
        
        # Optional: check if the command succeeded
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Experiment failed for $($file.Name)"
            # Uncomment the next line if you want the script to stop on error
            # break 
        }
    }
}

Write-Host "`nAll experiments completed." -ForegroundColor Green