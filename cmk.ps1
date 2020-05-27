<#
.SYNOPSIS
    CMake Helper script for FastFourieTransformTests package

.DESCRIPTION
    A script for simplifying the build of the FastFourieTransformTests package
    via CMake.

.PARAMETER BuildType
    Selects the value of CMAKE_BUILD_TYPE that should be used.

.PARAMETER Target
    Specifies the target name to pass to the CMake build command.

.PARAMETER Prefix
    Selects the value of CMAKE_INSTALL_PREFIX that should be used.

.PARAMETER DirForBuild
    Selects the directory to do the CMake build in.

.PARAMETER Clean
    Cleans the build directory before running CMake.

.PARAMETER Install
    Adds a target invocation to the CMake build command.

.EXAMPLE
    cmk.ps1
    Simple no argument build doesn't clean or install.

.EXAMPLE
    cmk.ps1 -c -i -b Release -p C:\Users\Foo\Bar foo
    Advanced build that cleans and builds in a custom directory foo.
    Also installs the output intp C:\Users\Foo\Bar.

.NOTES
    Author: Daniel Damiani
    Last Edit: 2020-05-25
    Version 1.0 - initial release of cmk.ps1

#>
[CmdletBinding()]
Param(
    [Parameter(Mandatory=$false, HelpMessage="The build type")]
    [ValidateSet("Debug", "Release", "RelWithDebInfo", "MinSizeRel")]
    [String]
    $BuildType = "Release",

    [Parameter(Mandatory=$false)]
    [String]
    $Target = "install",

    [Parameter(Mandatory=$false)]
    [String]
    $Prefix = "$pwd\install",

    [Parameter(Mandatory=$false, Position=0)]
    [String]
    $DirForBuild = "$pwd\build",

    [Parameter(Mandatory=$false)]
    [Switch]
    $Clean,

    [Parameter(Mandatory=$false)]                                                                                           [Switch]                                                                                                                $Install
)

$ErrorActionPreference = "Stop"

Function Add-PathVariable {
    param (
        [string]$addPath
    )
    if (Test-Path $addPath){
        $regexAddPath = [regex]::Escape($addPath)
        $arrPath = $env:Path -split ';' | Where-Object {$_ -notMatch "^$regexAddPath\\?"}
        $env:Path = ($arrPath + $addPath) -join ';'
    }
}

echo "$BuildType $DirForBuild"

# source the setup_env script if it exists
if ((Test-Path "$PSScriptRoot\setup_env.ps1" -PathType Leaf)) {
    . "$PSScriptRoot\setup_env.ps1"
}

if ($Clean) {
    Remove-Item -Force -Recurse -Path $DirForBuild
}

# create the build directory
[void](New-Item -ItemType Directory -Force -Path $DirForBuild)

# change to the build directory
Push-Location -Path $DirForBuild

cmake -DCMAKE_INSTALL_PREFIX="$Prefix" "$PSScriptRoot"
if ($?) {
    if($Install) {
        cmake --build . --config "$BuildType" --target "$Target"
        # if installing add it to the path
        Add-PathVariable "$Prefix/bin"
    } else {
        cmake --build . --config "$BuildType"
    }
}

# change back to the original directory
Pop-Location


