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

1. Select input mode (single file vs. multiple isotopes)
2. Choose files from a folder or enter file paths manually
3. View the full spectrum
4. Select channels for background subtraction
5. Review Gaussian fits and uncertainty analysis
6. Optionally convert channels to energy using the formula Channel = a + b*Energy
7. Get a summary of all peak positions

### Output

The tool generates:
- Spectrum plots (.png)
- Gaussian fit visualizations with uncertainty bands
- Processed data files (.csv)
- Fit results with detailed uncertainty estimates (.csv)

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

The peak position uncertainty is determined by combining three independent error sources using standard error propagation methodology. The final uncertainty is calculated by adding these three terms in quadrature:

```
final_err = sqrt(statistical_err² + window_variation_err² + channel_discretization_err²)
```

The three error components are:

1. **Statistical error**: The formal uncertainty derived from the covariance matrix of the Gaussian fit. This represents the statistical uncertainty inherent in fitting a mathematical model to experimental data.

2. **Window variation error**: This error is calculated by varying the selection window endpoints by ±5% of the window size and performing 5 independent Gaussian fits. The standard deviation of these peak positions provides an estimate of how sensitive the measurement is to the specific window selection.

3. **Channel discretization error**: A fixed value of 0.5 channels that accounts for the inherent uncertainty due to the discrete nature of the channel data. Since data is binned into integer channels, there is a minimum positional uncertainty of half a channel width.

By combining these three error sources, we obtain a comprehensive uncertainty estimate that accounts for:
- The statistical uncertainty of the fitting process
- The systematic effects of window selection
- The inherent limitations of discretized data

This approach follows standard error propagation principles for independent variables and ensures that all significant sources of uncertainty are properly accounted for in the final result.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.