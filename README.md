# Gamma Radiation Spectrum Analyzer

A Python tool for analyzing gamma radiation spectra, particularly useful for neutron activation analysis experiments. This tool processes CSV files containing radiation spectrum data, identifies peaks, and performs Gaussian fitting to accurately determine peak positions.

## Features

- Process multiple spectrum files in batch mode
- Interactive peak selection with background subtraction
- Gaussian curve fitting for accurate peak center determination
- Comprehensive uncertainty estimation using proper error propagation
- Detailed visualization of spectra and fitted peaks
- Summary of all peak positions across multiple isotopes
- Automatic output organization
- Channel to energy conversion with the form Channel = a + b*Energy
- Save user inputs to JSON for reproducible analysis
- Load previous analysis settings from JSON files
- Detailed chi-square statistics for goodness of fit evaluation

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/histogram_analyzer.git
   cd histogram_analyzer
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the program by executing:

```
python histogram_analyzer.py
```

### Input Data Format

The tool accepts CSV files containing radiation spectrum data with these structures:
- Files with "Channel Data:" marker followed by channel data
- Files with headers containing "channel" and "counts"
- Simple CSV files with minimal headers

The application will intelligently try to interpret various file formats.

### Processing Modes

1. **Single File Mode**: Process one spectrum file with multiple peaks
2. **Multiple Isotopes Mode**: Process multiple files representing different isotopes

### Workflow

1. Optionally load saved settings from a previous analysis
2. Select input mode (single file vs. multiple isotopes)
3. Choose files from a folder or enter file paths manually
4. View the full spectrum
5. Select channels for background subtraction
6. Review Gaussian fits, uncertainty analysis, and chi-square statistics
7. Optionally convert channels to energy using the formula Channel = a + b*Energy
8. Get a summary of all peak positions
9. Save all inputs and settings to JSON for reproducibility

### Output

The tool generates:
- Spectrum plots (.png)
- Gaussian fit visualizations with uncertainty bands
- Processed data files (.csv)
- Fit results with detailed uncertainty estimates (.csv)
- JSON files containing all user inputs for reproducibility (.json)

## Examples

### Basic Workflow

1. Run the program:
   ```
   python histogram_analyzer.py
   ```

2. Select processing mode (1 for single file or 2 for multiple isotopes)

3. Browse to your data files

4. For each peak, provide start and end channels for background subtraction

5. Review the results and plots

## Dependencies

- pandas: Data handling and CSV processing
- matplotlib: Visualization
- numpy: Numerical operations
- scipy: Signal processing and curve fitting

## Uncertainty Calculation

The peak position uncertainty is determined using multiple independent error sources. The calculation involves these steps:

1. Calculate individual error components
2. Combine the first three components in quadrature to get a combined error
3. Take the maximum of this combined error and a width-based estimate

The final formula is:
```
combined_err = sqrt(statistical_err² + window_variation_err² + channel_discretization_err²)
final_err = max(combined_err, width_based_err)
```

The four error components are:

1. **Statistical error**: The formal uncertainty derived from the covariance matrix of the Gaussian fit. This represents the statistical uncertainty inherent in fitting a mathematical model to experimental data.

2. **Window variation error**: This error is calculated by varying the selection window endpoints by ±5% of the window size and performing 1000 independent Gaussian fits (configurable via the WINDOW_VARIATION_RUNS parameter). The standard deviation of these peak positions provides an estimate of how sensitive the measurement is to the specific window selection.

3. **Channel discretization error**: A fixed value of 0.5 channels that accounts for the inherent uncertainty due to the discrete nature of the channel data. Since data is binned into integer channels, there is a minimum positional uncertainty of half a channel width.

4. **Width-based error**: A simple heuristic estimate calculated as 10% of the peak width (sigma). This provides a fallback that scales with the width of the peak.

> **⚠️ IMPORTANT NOTE ON UNCERTAINTY ESTIMATION ⚠️**
>
> **The uncertainty calculation implemented here is a relatively crude estimate and may need adjustment for your specific experimental conditions. Users are encouraged to customize the uncertainty calculation by:**
>
> **- Applying different scaling factors to individual error components**
> **- Changing the window variation ratio (currently set at ±5%)**
> **- Using different combination methods (maximum, quadrature sum, linear sum)**
> **- Adding additional error terms specific to your experimental setup**
>
> <span style="color:red">**USERS ARE STRONGLY ENCOURAGED TO EXPERIMENT WITH THESE METHODS TO FIND THE BEST APPROACH FOR THEIR SPECIFIC DATA AND EXPERIMENTAL NEEDS!**</span>
>
> **Uncertainty estimation is inherently experimental-dependent, and the method implemented here should be considered a starting point rather than a definitive solution.**

By using the maximum of the combined statistical error and the width-based estimate, we ensure a comprehensive uncertainty value that:
- Accounts for all significant error sources
- Scales appropriately with peak width
- Follows standard error propagation principles
- Never underestimates the true uncertainty

This approach provides a reasonable baseline for uncertainty estimation in typical neutron activation analysis experiments.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.