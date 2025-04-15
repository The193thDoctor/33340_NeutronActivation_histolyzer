# Detailed Usage Guide

## Input Data Format

This tool accepts CSV files containing radiation spectrum data. It's designed to be flexible and handle various formats commonly used in radiation spectroscopy:

1. **Standard Format**: Files with a "Channel Data:" marker followed by columns for Channel, Energy, and Counts.
2. **Simple Format**: CSV files with headers for Channel and Counts (Energy will be derived from Channel).
3. **Flexible Header Detection**: The tool will search for headers containing "channel" and "counts" terms.

## Processing Modes

The program offers two main modes of operation:

### 1. Single File Mode
Process one spectrum file with multiple peaks:
- Suitable for analyzing one spectrum with multiple peaks of interest
- Generates labeled plots with all detected peaks

### 2. Multiple Isotopes Mode
Process multiple files representing different isotopes:
- Batch processing of multiple CSV files
- Organizes outputs by isotope name
- Generates summary of all peaks across all isotopes

## Detailed Workflow

### 1. File Selection

When the program starts, you'll be asked to choose a processing mode:
```
Do you want to process (1) a single file or (2) multiple isotopes? (1/2):
```

You can either:
- Select files from a folder: The program will show all CSV files in the folder
- Enter file paths manually: Type the full path to your spectrum file

### 2. Spectrum Visualization

For each file, the program will:
- Show basic statistics about the loaded data
- Display the full spectrum as a histogram
- Ask how many peaks you want to analyze

### 3. Peak Analysis

For each peak:
1. You'll need to select two channels to define the background region:
   ```
   Enter starting channel for Peak: 
   Enter ending channel for Peak:
   ```
   - Choose channels that bracket your peak of interest
   - The region should include enough background on both sides of the peak

2. The program will:
   - Subtract a linear background between your chosen points
   - Fit a Gaussian curve to the background-subtracted peak
   - Calculate the peak center and its uncertainty through multiple methods

### 4. Output and Results

For each analyzed peak, the program generates:

1. **Visualizations**:
   - Full spectrum plot
   - Background subtraction plot
   - Gaussian fit with uncertainty band
   - Residuals analysis

2. **Data Files**:
   - `[element]_processed.csv`: The processed spectrum data
   - `[element]_fit_results.csv`: Detailed fitting parameters and uncertainties

3. **Summary Statistics**:
   - A table of all peak positions with uncertainties
   - Labels for each peak for easy reference

## Example Session

```
=== Radiation Spectrum Analysis Tool ===

Do you want to process (1) a single file or (2) multiple isotopes? (1/2): 1

Do you want to select from a folder containing CSV files? (y/n): y

Enter the path to the folder containing CSV files: /path/to/data

Found 3 CSV files:
1. cobalt60.csv
2. sodium22.csv
3. background.csv

Enter the number of the file to process: 1

Suggested element name based on filename: cobalt60
Press Enter to accept this name, or type a different name:
Element: 

Successfully loaded Co-60 data with 1024 channels
Max counts: 4872 at channel 346

How many peaks would you like to process for cobalt60? 2

=== Processing cobalt60 Peak 1 ===

You need to select two channels to define the background region.
Enter starting channel for cobalt60 Peak 1: 310
Enter ending channel for cobalt60 Peak 1: 380

[Analysis proceeds with plots and results...]

=== Processing cobalt60 Peak 2 ===

You need to select two channels to define the background region.
Enter starting channel for cobalt60 Peak 2: 450
Enter ending channel for cobalt60 Peak 2: 500

[Analysis proceeds with plots and results...]

Do you want to create a spectrum with labeled peak positions? (y/n): y

[Final labeled spectrum is displayed with both peaks marked]

======================================================================
                SUMMARY OF ALL PEAK LOCATIONS AND UNCERTAINTIES       
======================================================================
Peak Labels (for reference only):
1: cobalt60_peak1
2: cobalt60_peak2

Peak Locations (channels):
[345.82, 478.15]

Peak Uncertainties (channels, 1σ):
[0.5, 0.68]
======================================================================
              Copy-paste the above lists for your records!            
======================================================================
```

## Tips for Best Results

1. **Peak Selection**:
   - Choose peak regions wide enough to include sufficient background on both sides
   - Make sure the background regions you select are relatively flat
   - If a peak has a complex structure, try analyzing it in segments

2. **Multiple Peaks**:
   - For isotopes with multiple peaks, create a labeled spectrum to visualize all peaks together
   - Compare peak positions with reference values for isotope identification

3. **Uncertainty Analysis**:
   - The program calculates uncertainty through multiple methods and uses the most conservative
   - For publication-quality results, note the reduced chi-square value of the fits
   - Peaks with high channel counts will typically have lower relative uncertainty