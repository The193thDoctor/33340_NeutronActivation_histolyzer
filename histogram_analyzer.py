import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import json
import datetime
import re
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy import stats

def read_spectrum_data(filepath):
    """
    Read spectroscopy data from CSV file and extract histogram data.
    1
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with Channel, Energy, and Counts columns
    """
    
    # Read the file content as text to find where channel data starts
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Find the line that has "Channel Data:"
    channel_data_start = 0
    for i, line in enumerate(lines):
        if "Channel Data:" in line:
            channel_data_start = i + 2  # +2 to skip header row
            break
    
    # If "Channel Data:" wasn't found, try to infer structure
    if channel_data_start == 0:
        # Look for a line that might contain column headers like "Channel" and "Counts"
        for i, line in enumerate(lines):
            if ("channel" in line.lower() or "ch" in line.lower()) and "count" in line.lower():
                channel_data_start = i + 1
                break
        
        # If still not found, assume it's a simple CSV with headers in the first row
        if channel_data_start == 0:
            channel_data_start = 1
    
    # Read the channel data into a pandas DataFrame
    try:
        data = pd.read_csv(filepath, header=None, skiprows=channel_data_start, 
                           names=['Channel', 'Energy', 'Counts'])
        
        # Convert counts to numeric, handling empty values
        data['Counts'] = pd.to_numeric(data['Counts'], errors='coerce')
        
        # If Energy column is empty, create a placeholder
        if data['Energy'].isna().all():
            data['Energy'] = data['Channel']
    
    except Exception as e:
        print(f"Error reading CSV format, trying alternative: {e}")
        # Try a simpler approach with just two columns
        try:
            data = pd.read_csv(filepath, header=None, skiprows=channel_data_start, 
                              names=['Channel', 'Counts'])
            data['Energy'] = data['Channel']  # Create placeholder Energy column
            data = data[['Channel', 'Energy', 'Counts']]  # Reorder columns
        except Exception as e2:
            print(f"Failed to read file: {e2}")
            raise
    
    # Convert counts to numeric, handling empty values
    data['Counts'] = pd.to_numeric(data['Counts'], errors='coerce')
    
    return data

def ensure_output_folder(input_folder=None):
    """
    Create output folder if it doesn't exist.
    
    Args:
        input_folder: Optional input folder path. If provided, output folder will be created inside it.
        
    Returns:
        Path to the output folder
    """
    if input_folder:
        output_folder = os.path.join(input_folder, "output")
    else:
        output_folder = "output"
        
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")
    
    return output_folder

def plot_histogram(data, element_name, title=None, file_prefix=None, output_folder=None, peak_labels=None):
    """
    Plot the histogram data.
    
    Args:
        data: DataFrame with channels and counts
        element_name: Name of the element being analyzed
        title: Optional custom title
        file_prefix: Optional prefix for the output filename
        output_folder: Optional output folder path
        peak_labels: Optional list of (channel, label) tuples to mark on the plot
    
    Returns:
        List of detected peak positions
    """
    plt.figure(figsize=(12, 6))
    plt.bar(data['Channel'], data['Counts'], width=1.0, color='blue', alpha=0.7)
    plt.xlabel('Channel')
    plt.ylabel('Counts')
    
    # Set x-axis to start from 0 and end at the highest channel
    plt.xlim(0, data['Channel'].max())
    
    if title:
        plt.title(title)
    else:
        plt.title(f"{element_name} Radiation Spectrum")
    
    plt.grid(True, alpha=0.3)
    
    # If peak_labels is provided, use those for labeling
    if peak_labels:
        # Try to use peak_heights if provided as a parameter
        peak_heights = getattr(data, 'peak_heights', None)
        
        for i, (channel, label) in enumerate(peak_labels):
            # If we have peak heights data, use it
            if isinstance(peak_heights, dict) and str(i) in peak_heights:
                label_height = peak_heights[str(i)]
                # Ensure label is visible
                label_height = max(label_height, data['Counts'].max() * 0.25)
            else:
                # Fallback to actual count or 90% of max
                label_height = data.loc[data['Channel'] == channel, 'Counts'].values[0] if channel in data['Channel'].values else data['Counts'].max() * 0.9
            
            plt.annotate(label,
                         xy=(channel, label_height),
                         xytext=(0, 10),
                         textcoords='offset points',
                         ha='center')
    
    plt.tight_layout()
    
    # Use file_prefix if provided, otherwise use element_name
    filename = f"{file_prefix or element_name}_spectrum.png"
    
    # Save to output folder if specified
    if output_folder:
        output_path = os.path.join(output_folder, filename)
    else:
        output_path = filename
    
    plt.savefig(output_path, dpi=300)
    plt.show()
    
    print(f"Spectrum saved to: {output_path}")
    
    # Only detect peaks if we didn't have explicit labels
    if not peak_labels:
        # Just for compatibility, return detected peaks
        peaks, _ = find_peaks(data['Counts'], height=0.5*data['Counts'].max(), distance=20)
        return peaks
    else:
        # Return the channels from our labels
        return [channel for channel, _ in peak_labels]

def gaussian(x, a, mu, sigma, c):
    """Gaussian function with offset."""
    return a * np.exp(-(x - mu)**2 / (2 * sigma**2)) + c

def find_peak_with_background_subtraction_and_fit(data, start_ch, end_ch, element_name, peak_id=None, uncertainty_factor=1.0, output_folder=None):
    """
    Find peaks after subtracting linear background between two points,
    then fit a Gaussian to get more accurate peak position with standard uncertainty.
    
    Args:
        data: DataFrame with channels and counts
        start_ch: Starting channel for background subtraction
        end_ch: Ending channel for background subtraction
        element_name: Name of the element being analyzed
        peak_id: Optional identifier for distinguishing multiple peaks
        uncertainty_factor: Factor to scale uncertainty estimates (default=1.0)
        output_folder: Optional output folder path
    
    Returns:
        Fitted peak parameters (center, width, amplitude, uncertainty)
    """
    # Generate file prefix for outputs - peak_id is now just the number
    file_prefix = f"{element_name}_peak{peak_id}" if peak_id else element_name
    
    # Get counts at the specified channels
    start_count = data.loc[data['Channel'] == start_ch, 'Counts'].values[0]
    end_count = data.loc[data['Channel'] == end_ch, 'Counts'].values[0]
    
    print(f"Point 1: Channel {start_ch}, Counts {start_count}")
    print(f"Point 2: Channel {end_ch}, Counts {end_count}")
    
    # Create a copy of the data for the range we're interested in
    mask = (data['Channel'] >= start_ch) & (data['Channel'] <= end_ch)
    range_data = data[mask].copy()
    
    # Calculate linear interpolation
    channels = range_data['Channel'].values
    m = (end_count - start_count) / (end_ch - start_ch)
    b = start_count - m * start_ch
    background = m * channels + b
    
    # Subtract background
    range_data['Background'] = background
    range_data['Subtracted'] = range_data['Counts'] - range_data['Background']
    
    # Find the approximate maximum to use as initial guess for Gaussian fit
    max_idx = range_data['Subtracted'].idxmax()
    max_channel = range_data.loc[max_idx, 'Channel']
    max_subtracted = range_data.loc[max_idx, 'Subtracted']
    
    # Fit Gaussian to the background-subtracted data
    x_data = range_data['Channel'].values
    y_data = range_data['Subtracted'].values
    
    # Initial parameter guesses [amplitude, mean, sigma, constant]
    p0 = [max_subtracted, max_channel, (end_ch - start_ch)/10, 0]
    
    try:
        # Fit the Gaussian
        popt, pcov = curve_fit(gaussian, x_data, y_data, p0=p0)

        # Extract parameters
        amplitude, mu, sigma, const = popt

        # Calculate parameter uncertainties from covariance matrix
        perr = np.sqrt(np.diag(pcov))

        # Apply uncertainty scale factor (default is 1.0 for standard error)
        perr = perr * uncertainty_factor

        # Calculate correlation matrix (using original covariance)
        correlation = pcov / np.outer(np.sqrt(np.diag(pcov)), np.sqrt(np.diag(pcov)))

        # Generate fitted curve
        y_fit = gaussian(x_data, *popt)

        # Calculate residuals and chi-square
        residuals = y_data - y_fit

        # Calculate Poisson-based uncertainties for each point
        # Standard error for count data is approximately sqrt(N)
        poisson_errors = np.sqrt(np.abs(y_fit))

        # Use the larger of the Poisson error or the residual for each point
        # to account for both statistical and potential systematic uncertainties
        combined_errors = np.maximum(poisson_errors, np.abs(residuals))

        # Recalculate chi-square with these conservative errors
        chi_square = np.sum((residuals / combined_errors)**2)
        dof = len(y_data) - len(popt)  # degrees of freedom
        reduced_chi_square = chi_square / dof
        p_value = 1 - stats.chi2.cdf(chi_square, dof)

        # Calculate confidence interval for peak position (1-sigma)
        conf_interval = perr[1]

        # Additional uncertainty estimation methods
        
        # Width-based estimate - 10% of the peak width as a simple heuristic
        width_based_err = sigma * 0.1  # 10% of the peak width
        
        # Estimate from peak position variation by varying window size
        # Vary the left and right window points by a random amount within 5% of the window size
        window_variations = []
        window_size = end_ch - start_ch
        
        # Using a fixed seed would defeat the randomness, so don't set one
        for i in range(5):
            # Vary window endpoints by random amounts within 5% of window size
            # Make sure we're generating new random values each time
            var_window_left = max(0, start_ch + int(np.random.uniform(-0.05, 0.05) * window_size))
            var_window_right = min(max(channels), end_ch + int(np.random.uniform(-0.05, 0.05) * window_size))
            
            # Create a mask for the varied window
            var_mask = (range_data['Channel'] >= var_window_left) & (range_data['Channel'] <= var_window_right)
            var_range_data = range_data[var_mask].copy()
            
            # Skip if we don't have enough data points
            if len(var_range_data) < 10:
                continue
                
            var_channels = var_range_data['Channel'].values
            
            # Calculate linear background using the varied window endpoints
            var_start_count = data.loc[data['Channel'] == var_window_left, 'Counts'].values[0]
            var_end_count = data.loc[data['Channel'] == var_window_right, 'Counts'].values[0]
            
            var_m = (var_end_count - var_start_count) / (var_window_right - var_window_left)
            var_b = var_start_count - var_m * var_window_left
            var_background = var_m * var_channels + var_b

            # Subtract new background
            var_subtracted = var_range_data['Counts'].values - var_background

            # Find the approximate maximum to use as initial guess for Gaussian fit
            var_peak_idx = np.argmax(var_subtracted)
            var_max_channel = var_channels[var_peak_idx]
            var_max_subtracted = var_subtracted[var_peak_idx]

            try:
                # Initial parameter guesses [amplitude, mean, sigma, constant]
                var_width_guess = (var_window_right - var_window_left) / 10
                var_p0 = [var_max_subtracted, var_max_channel, var_width_guess, 0]

                # Fit the Gaussian - using all data in the varied window
                var_x_data = var_channels
                var_y_data = var_subtracted
                
                var_popt, _ = curve_fit(gaussian, var_x_data, var_y_data, p0=var_p0)

                # Extract the peak center (mu parameter)
                var_refined_peak = var_popt[1]
                window_variations.append(var_refined_peak)
            except Exception:
                # Fallback if fit fails - just use the simple maximum
                window_variations.append(var_max_channel)

        # Calculate standard deviation of peak positions from window variations
        window_var_err = np.std(window_variations) if window_variations else sigma * 0.05  # default fallback

        # Channel discretization error (half a channel)
        channel_err = 0.5

        # Combine errors in quadrature (for independent error sources)
        combined_err = np.sqrt(perr[1]**2 + window_var_err**2 + channel_err**2)
        
        # Use the maximum of the combined error and the width-based error
        final_err = max(combined_err, width_based_err)

        print(f"\nUncertainty Estimation Components:")
        print(f"Statistical (from fit): {perr[1]:.4f} channels")
        print(f"Window variation: {window_var_err:.4f} channels")
        print(f"Channel discretization: {channel_err:.4f} channels")
        print(f"Width-based (10% of sigma): {width_based_err:.4f} channels")
        print(f"Combined (quadrature): {combined_err:.4f} channels")
        print(f"Final uncertainty (max): {final_err:.4f} channels")

        # Plot the results
        plt.figure(figsize=(12, 12))

        # Plot 1: Original data with linear background
        plt.subplot(311)
        plt.bar(range_data['Channel'], range_data['Counts'], width=1.0,
                color='blue', alpha=0.7, label='Original Data')
        plt.plot(range_data['Channel'], range_data['Background'], 'r-',
                linewidth=2, label='Linear Background')
        plt.xlabel('Channel')
        plt.ylabel('Counts')

        # Use peak_id in title if provided
        peak_title = f"{element_name}"
        if peak_id:
            peak_title = f"{element_name} Peak {peak_id}"

        plt.title(f'{peak_title} Spectrum with Linear Background')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 2: Background-subtracted data with Gaussian fit
        plt.subplot(312)
        plt.bar(range_data['Channel'], range_data['Subtracted'], width=1.0,
                color='green', alpha=0.7, label='Background Subtracted')
                
        # Generate a smooth curve for plotting using dense points
        x_dense = np.linspace(min(x_data), max(x_data), 500)
        y_dense = gaussian(x_dense, *popt)
                
        plt.plot(x_dense, y_dense, 'r-', linewidth=2,
                label=f'Gaussian Fit (μ={mu:.2f}±{final_err:.2f})')

        # Add shaded region to indicate uncertainty
        plt.axvspan(mu - final_err, mu + final_err,
                   alpha=0.2, color='red', label='Peak Uncertainty Interval')
        plt.axvline(x=mu, color='k', linestyle='--')
        plt.xlabel('Channel')
        plt.ylabel('Counts (Background Subtracted)')
        plt.title(f'{peak_title} with Gaussian Fit & Uncertainty')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 3: Residuals
        plt.subplot(313)
        
        # Ensure x_data and residuals have the same length
        if len(x_data) != len(residuals):
            # Use channel numbers as x-axis instead
            plt.bar(range(len(residuals)), residuals, width=1.0, color='purple', alpha=0.7)
            plt.fill_between(range(len(residuals)), -combined_errors, combined_errors, color='gray', alpha=0.3,
                             label='Estimated Error Band')
            # Set x-axis to match the original channel range
            plt.xlim(0, len(residuals)-1)
            plt.xticks([])  # Hide x ticks as they don't directly correspond to channels
        else:
            plt.bar(x_data, residuals, width=1.0, color='purple', alpha=0.7)
            plt.fill_between(x_data, -combined_errors, combined_errors, color='gray', alpha=0.3,
                             label='Estimated Error Band')
        plt.axhline(y=0, color='k', linestyle='-')
        plt.xlabel('Channel')
        plt.ylabel('Residuals (Data - Fit)')
        plt.title(f'{peak_title} Fit Residuals')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Generate filename based on element name and peak ID
        plt.tight_layout()

        # Save to output folder if specified
        filename = f'{file_prefix}_gaussian_fit.png'
        if output_folder:
            output_path = os.path.join(output_folder, filename)
        else:
            output_path = filename

        plt.savefig(output_path, dpi=300)
        plt.show()

        print(f"Gaussian fit plot saved to: {output_path}")

        # Print detailed statistical information
        print(f"\nDetailed Analysis for {peak_title}:")
        print(f"Peak Center: {mu:.4f} ± {final_err:.4f} channels (1σ)")
        print(f"FWHM: {2.355 * sigma:.4f} ± {2.355 * perr[2]:.4f} channels")
        print(f"Reduced Chi-square: {reduced_chi_square:.4f} (DoF: {dof})")

        # Calculate total peak height (amplitude + background at peak center)
        # Background at peak center using m and b from earlier linear interpolation
        peak_height = amplitude + (m * mu + b)
        
        # Return fit results
        fit_results = {
            'element': element_name,
            'peak_id': peak_id,   # Store peak ID in results
            'center': mu,
            'center_err': final_err,  # 1-sigma error
            'sigma': sigma,
            'sigma_err': perr[2],
            'amplitude': amplitude,
            'amplitude_err': perr[0],
            'constant': const,
            'constant_err': perr[3],
            'peak_height': peak_height,  # Total height including background
            'fwhm': 2.355 * sigma,  # FWHM = 2.355 * sigma for Gaussian
            'fwhm_err': 2.355 * perr[2],
            'chi_square': chi_square,
            'reduced_chi_square': reduced_chi_square,
            'p_value': p_value,
            'correlation_matrix': correlation,
            'residuals': residuals,
            'stat_err': perr[1],
            'window_var_err': window_var_err,
            'channel_err': channel_err,
            'width_based_err': width_based_err,
            'combined_err': combined_err
        }

        return fit_results
        
    except Exception as e:
        print(f"Error in Gaussian fitting: {e}")
        # If fitting fails, return the simple maximum with a generous uncertainty
        # Calculate total peak height (raw subtracted value + background at that position)
        peak_height = max_subtracted + (m * max_channel + b)
        
        return {
            'element': element_name,
            'peak_id': peak_id,
            'center': max_channel,
            'center_err': (end_ch - start_ch) / 10,  # Conservative fallback
            'amplitude': max_subtracted,
            'amplitude_err': np.sqrt(max_subtracted) * 2,  # Poisson error × 2
            'peak_height': peak_height
        }

def save_user_input(input_data, filename=None, output_folder=None):
    """
    Save user input data to a JSON file for future reference.
    
    Args:
        input_data: Dictionary containing user inputs
        filename: Optional custom filename (without extension)
        output_folder: Optional output folder path
    
    Returns:
        Path to the saved file
    """
    # Add timestamp
    input_data['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create filename based on element name or default
    if not filename:
        if 'element_name' in input_data:
            filename = f"{input_data['element_name']}_input"
        else:
            filename = "spectrum_config"
    
    filename = f"{filename}.json"
    
    # Save to output folder if specified
    if output_folder:
        output_path = os.path.join(output_folder, filename)
    else:
        output_path = filename
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(input_data, f, indent=4)
    
    print(f"User inputs saved to: {output_path}")
    return output_path

def load_user_input_from_file(filepath):
    """
    Load user input data from a JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Dictionary containing user inputs
    """
    try:
        with open(filepath, 'r') as f:
            input_data = json.load(f)
        print(f"Loaded user inputs from: {filepath}")
        return input_data
    except Exception as e:
        print(f"Error loading input file: {e}")
        return None

def process_peak(data, element_name, peak_id=None, output_folder=None, input_data={}):
    """
    Process a single peak with user input for channel selection.
    
    Args:
        data: DataFrame with channels and counts
        element_name: Name of the element being analyzed
        peak_id: Optional identifier for distinguishing multiple peaks
        output_folder: Optional output folder path
        input_data: Optional pre-loaded input data
    
    Returns:
        Fit results for this peak
    """
    # Create a formatted title string based on peak_id
    peak_title = f"{element_name}"
    if peak_id:
        peak_title = f"{element_name} Peak {peak_id}"
    
    print(f"\n=== Processing {peak_title} ===")
    
    # Generate input key for this peak
    input_key = f"{element_name}_peak{peak_id}" if peak_id else element_name
    
    # Check if we have pre-loaded input data for this peak
    if input_key in input_data:
        print(f"Using pre-loaded input data for {peak_title}")
        start_ch = input_data[input_key]['start_ch']
        end_ch = input_data[input_key]['end_ch']
        print(f"Start channel: {start_ch}, End channel: {end_ch}")
    else:
        # Query user for two channel locations
        print("\nYou need to select two channels to define the background region.")
        while True:
            try:
                start_ch = int(input(f"Enter starting channel for {peak_title}: "))
                if start_ch not in data['Channel'].values:
                    print(f"Warning: Channel {start_ch} not found in data. Available range: "
                          f"{data['Channel'].min()} to {data['Channel'].max()}")
                    continue
                
                end_ch = int(input(f"Enter ending channel for {peak_title}: "))
                if end_ch not in data['Channel'].values:
                    print(f"Warning: Channel {end_ch} not found in data. Available range: "
                          f"{data['Channel'].min()} to {data['Channel'].max()}")
                    continue
                
                if start_ch >= end_ch:
                    print("Error: Starting channel must be less than ending channel.")
                    continue
                
                break

            except ValueError:
                print("Error: Please enter valid integer channel numbers.")
                
        # Save this input to the user inputs dictionary
        input_data[input_key] = {
            'start_ch': start_ch,
            'end_ch': end_ch
        }
    
    # Always use standard 1-sigma uncertainty (no user prompt)
    uncertainty_factor = 1.0
    
    # Find peak with background subtraction and Gaussian fitting
    fit_results = find_peak_with_background_subtraction_and_fit(
        data, start_ch, end_ch, element_name, peak_id, uncertainty_factor, output_folder)
    
    # Generate output filenames based on element name and peak ID
    file_prefix = f"{element_name}_peak{peak_id}" if peak_id else element_name
    
    # Define output paths based on output folder
    if output_folder:
        processed_filename = os.path.join(output_folder, f"{file_prefix}_processed.csv")
        results_filename = os.path.join(output_folder, f"{file_prefix}_fit_results.csv")
    else:
        processed_filename = f"{file_prefix}_processed.csv"
        results_filename = f"{file_prefix}_fit_results.csv"
    
    # Save processed data and results
    data.to_csv(processed_filename, index=False)
    print(f"\nProcessed data saved to '{processed_filename}'")
    
    # Save fit results to a separate file
    if 'correlation_matrix' in fit_results:
        corr_matrix = fit_results.pop('correlation_matrix')
        residuals = fit_results.pop('residuals')
        pd.DataFrame([fit_results]).to_csv(results_filename, index=False)
        print(f"Fit results saved to '{results_filename}'")
    
    print(f"\n{peak_title} analysis complete!")
    
    return fit_results

def process_multiple_peaks_one_isotope(data, element_name, output_folder=None, input_data={}):
    """
    Process multiple peaks for a single isotope.
    
    Args:
        data: DataFrame with channels and counts
        element_name: Name of the element being analyzed
        output_folder: Optional output folder path
        input_data: Optional pre-loaded input data
    
    Returns:
        Dictionary containing peak results and additional information
    """
    # Store element name
    input_data['element_name'] = element_name
    
    # Store output folder
    if output_folder:
        input_data['output_folder'] = output_folder
    # Plot the initial spectrum without peak labels
    title = f"{element_name} Radiation Spectrum"
    plot_histogram(data, element_name, title=title, output_folder=output_folder)
    
    # Check if number of peaks is in the input data
    if 'num_peaks' in input_data:
        num_peaks = input_data['num_peaks']
        print(f"\nUsing pre-loaded number of peaks: {num_peaks}")
    else:
        # Ask how many peaks to process
        while True:
            try:
                num_peaks = int(input(f"\nHow many peaks would you like to process for {element_name}? "))
                if num_peaks <= 0:
                    print("Error: Please enter a positive number.")
                    continue
                break
            except ValueError:
                print("Error: Please enter a valid integer.")
                
        # Save number of peaks to input data
        input_data['num_peaks'] = num_peaks
    
    # Process each peak
    all_results = {}
    peak_locations = []
    peak_uncertainties = []
    
    for i in range(num_peaks):
        # For a single peak, use None as peak_id. For multiple peaks, use numbers.
        peak_id = f"{i+1}" if num_peaks > 1 else None
        fit_results = process_peak(data, element_name, peak_id, output_folder, input_data)
        all_results[peak_id or 'main'] = fit_results
        
        # Store peak locations and uncertainties, converting numpy values to Python float
        peak_locations.append(float(fit_results['center']))
        peak_uncertainties.append(float(fit_results['center_err']))
    
    # Ask if user wants to create a labeled spectrum with the fitted peak positions
    if peak_locations:
        # Check if label_peaks preference is in input data
        if 'label_peaks' in input_data:
            label_peaks = input_data['label_peaks']
            print(f"\nUsing pre-loaded preference for labeled peaks: {'Yes' if label_peaks else 'No'}")
        else:
            label_peaks = input("\nDo you want to create a spectrum with labeled peak positions? (y/n): ").lower().startswith('y')
            # Save preference to input data
            input_data['label_peaks'] = label_peaks
        
        if label_peaks:
            # Check if energy conversion is in the input data
            use_energy = False
            a = None
            b = None
            a_err = None
            b_err = None
            
            if 'energy_conversion' in input_data:
                use_energy = True
                energy_conversion = input_data['energy_conversion']
                a = energy_conversion['a']
                a_err = energy_conversion['a_err']
                b = energy_conversion['b']
                b_err = energy_conversion['b_err']
                print(f"\nUsing pre-loaded energy conversion parameters:")
                print(f"a = {a} ± {a_err}, b = {b} ± {b_err}")
            else:
                # Ask if user wants to convert to energy (MeV)
                use_energy = input("\nDo you want to convert the x-axis to energy (MeV)? (y/n): ").lower().startswith('y')
            
            if use_energy:
                # Only ask for parameters if we don't already have them from the config
                if a is None or b is None or a_err is None or b_err is None:
                    # Get conversion parameters (C = a + b*E)
                    print("\nPlease enter the parameters for the conversion: Channel = a + b*Energy(MeV)")
                    while True:
                        try:
                            a = float(input("Parameter a: "))
                            a_err = float(input("Uncertainty in a: "))
                            b = float(input("Parameter b: "))
                            b_err = float(input("Uncertainty in b: "))
                            
                            if b == 0:
                                print("Error: Parameter 'b' cannot be zero.")
                                continue
                            
                            break
                        except ValueError:
                            print("Error: Please enter valid numbers.")
                
                # Save energy conversion parameters
                input_data['energy_conversion'] = {
                    'a': a,
                    'a_err': a_err,
                    'b': b,
                    'b_err': b_err
                }
                
                # Create a copy of the data with energy axis
                data_with_energy = data.copy()
                # Convert channel to energy: E = (C-a)/b
                data_with_energy['Energy_MeV'] = (data_with_energy['Channel'] - a) / b
                
                # Convert peak locations to energy with propagated errors
                energy_peaks = []
                energy_uncertainties = []
                
                for loc, err in zip(peak_locations, peak_uncertainties):
                    # Convert channel to energy: E = (C-a)/b
                    energy = (loc - a) / b
                    
                    # Propagate uncertainties
                    # For function E = (C-a)/b, the uncertainty is:
                    # σ_E^2 = (∂E/∂C)^2 * σ_C^2 + (∂E/∂a)^2 * σ_a^2 + (∂E/∂b)^2 * σ_b^2
                    # where ∂E/∂C = 1/b, ∂E/∂a = -1/b, ∂E/∂b = -(C-a)/b^2

                    dE_dC = 1/b
                    dE_da = -1/b
                    dE_db = -(loc-a)/(b**2)
                    
                    energy_err = np.sqrt((dE_dC**2 * err**2) + 
                                         (dE_da**2 * a_err**2) + 
                                         (dE_db**2 * b_err**2))
                    
                    energy_peaks.append(energy)
                    energy_uncertainties.append(energy_err)
                
                # Create labels for each peak in energy units
                peak_labels = []
                for i, (energy, err) in enumerate(zip(energy_peaks, energy_uncertainties)):
                    peak_label = f"{energy:.3f}±{err:.3f} MeV"
                    # Find the corresponding channel
                    channel = peak_locations[i]
                    closest_channel = int(round(channel))
                    peak_labels.append((closest_channel, peak_label))
                
                # Create a new plot with the fitted peak positions labeled in energy
                plt.figure(figsize=(12, 6))
                plt.bar(data_with_energy['Energy_MeV'], data['Counts'], width=(data_with_energy['Energy_MeV'].max() - data_with_energy['Energy_MeV'].min())/len(data_with_energy), 
                       color='blue', alpha=0.7)
                
                plt.xlabel('Energy (MeV)')
                plt.ylabel('Counts')
                
                # Set x-axis to start from 0 MeV and show the full spectrum initially
                plt.xlim(0, data_with_energy['Energy_MeV'].max())
                
                # Show the plot to help user determine a good max energy value
                plt.title(f"{element_name} Radiation Spectrum (Initial View)")
                plt.tight_layout()
                plt.show()
                
                # Check if max_energy is in the input data
                if 'max_energy' in input_data:
                    max_energy = input_data['max_energy']
                    print(f"\nUsing pre-loaded maximum energy: {max_energy} MeV")
                else:
                    # Now ask for the maximum energy for the x-axis after showing the spectrum
                    print("\nAfter viewing the spectrum, you can now set a custom energy range.")
                    while True:
                        try:
                            max_energy = float(input("Enter the maximum energy (MeV) for the x-axis (or 0 to use the default): "))
                            if max_energy < 0:
                                print("Error: Maximum energy must be a non-negative number.")
                                continue
                            break
                        except ValueError:
                            print("Error: Please enter a valid number.")
                    
                    # Save max_energy to input data
                    input_data['max_energy'] = max_energy
                
                # Create the final plot with the fitted peak positions labeled in energy
                plt.figure(figsize=(12, 6))
                plt.bar(data_with_energy['Energy_MeV'], data['Counts'], width=(data_with_energy['Energy_MeV'].max() - data_with_energy['Energy_MeV'].min())/len(data_with_energy), 
                       color='blue', alpha=0.7)
                
                plt.xlabel('Energy (MeV)')
                plt.ylabel('Counts')
                
                # Set x-axis to start from 0 MeV and end at the user-specified energy
                # If user entered 0, use the maximum energy in the data
                if max_energy > 0:
                    plt.xlim(0, max_energy)
                else:
                    plt.xlim(0, data_with_energy['Energy_MeV'].max())
                
                title = f"{element_name} Radiation Spectrum (Fitted Peaks)"
                plt.title(title)
                
                plt.grid(True, alpha=0.3)
                
                # Add peak labels
                for i, (channel, label) in enumerate(peak_labels):
                    energy_value = (channel - a) / b
                    
                    # Get the peak_id (key in all_results dictionary)
                    peak_key = f"{i+1}" if num_peaks > 1 else 'main'
                    fit_result = all_results.get(peak_key, {})
                    
                    # Use fitted peak height if available
                    if 'peak_height' in fit_result:
                        label_height = fit_result['peak_height']
                        # For very small peaks, ensure the label is visible
                        label_height = max(label_height, data['Counts'].max() * 0.25)
                    else:
                        # Fallback to actual count or 90% of max
                        label_height = data.loc[data['Channel'] == channel, 'Counts'].values[0] if channel in data['Channel'].values else data['Counts'].max() * 0.9
                    
                    plt.annotate(label,
                                 xy=(energy_value, label_height),
                                 xytext=(0, 10),
                                 textcoords='offset points',
                                 ha='center')
                
                plt.tight_layout()
                
                # File name and save
                file_prefix = f"{element_name}_energy_labeled"
                if output_folder:
                    output_path = os.path.join(output_folder, f"{file_prefix}_spectrum.png")
                else:
                    output_path = f"{file_prefix}_spectrum.png"
                
                plt.savefig(output_path, dpi=300)
                plt.show()
                
                print(f"Energy spectrum saved to: {output_path}")
                
                # Print energy peak summary
                print("\n=== Peak Energies ===")
                for i, (energy, err) in enumerate(zip(energy_peaks, energy_uncertainties)):
                    peak_num = i+1 if len(energy_peaks) > 1 else ""
                    print(f"Peak {peak_num}: {energy:.4f} ± {err:.4f} MeV")
                
            else:
                # Create labels for each peak in channel units
                peak_labels = []
                
                # Also collect peak heights
                peak_heights = {}
                
                for i, (loc, err) in enumerate(zip(peak_locations, peak_uncertainties)):
                    peak_label = f"Ch: {loc:.1f}±{err:.1f}"
                    # Find the closest channel in the data
                    closest_channel = int(round(loc))
                    peak_labels.append((closest_channel, peak_label))
                    
                    # Store the peak height from fit results
                    peak_key = f"{i+1}" if num_peaks > 1 else 'main'
                    fit_result = all_results.get(peak_key, {})
                    if 'peak_height' in fit_result:
                        peak_heights[str(i)] = fit_result['peak_height']
                
                # Add peak heights to data for use in plotting
                data.peak_heights = peak_heights
                
                # Create a new plot with the fitted peak positions labeled
                title = f"{element_name} Radiation Spectrum (Fitted Peaks)"
                file_prefix = f"{element_name}_labeled"
                plot_histogram(data, element_name, title=title, file_prefix=file_prefix, 
                              output_folder=output_folder, peak_labels=peak_labels)
    
    # Save user inputs at the end, after all inputs have been collected
    save_user_input(input_data, element_name, output_folder)
    
    # Return all the results as well as peaks information
    return {
        'results': all_results,
        'element_name': element_name,
        'peak_locations': peak_locations,
        'peak_uncertainties': peak_uncertainties
    }

def list_csv_files(folder_path):
    """
    Lists all CSV files in the specified folder.
    
    Args:
        folder_path: Path to the folder containing CSV files
        
    Returns:
        List of CSV files with their full paths
    """
    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a valid directory.")
        return []
        
    csv_files = []
    for file in os.listdir(folder_path):
        if file.lower().endswith('.csv'):
            full_path = os.path.join(folder_path, file)
            if os.path.isfile(full_path):
                csv_files.append(full_path)
    
    return csv_files

def ask_for_file_selection():
    """
    Ask the user for file selection input.
    
    Returns:
        String containing the user's file selection
    """
    print("\nWhich files would you like to process?")
    print("Enter file numbers separated by commas (e.g., 1,3,5), or 'all' for all files:")
    return input("Selection: ").strip().lower()

def process_single_file(filepath, input_data=None, output_folder=None):
    """
    Process a single spectrum data file, with multiple peaks.
    
    Args:
        filepath: Path to the CSV file to process
        input_data: Optional dictionary containing pre-loaded input parameters
        output_folder: Optional output folder path
    
    Returns:
        Dictionary containing results for this file
    """
    # Initialize input_data if not provided
    if input_data is None:
        input_data = {}
    
    # File ID is the base filename (used as a key in configs)
    file_id = os.path.basename(filepath)
    
    # Get element name from file-specific config if available
    if file_id in input_data and 'element_name' in input_data[file_id]:
        element_name = input_data[file_id]['element_name']
        print(f"\nUsing pre-configured element name for {file_id}: {element_name}")
    # Otherwise check for global element_name
    elif 'element_name' in input_data:
        element_name = input_data['element_name']
        print(f"\nUsing element name from input data: {element_name}")
    else:
        # Suggest from filename
        filename_base = os.path.splitext(os.path.basename(filepath))[0]
        print(f"\nSuggested element name based on filename: {filename_base}")
        print("Press Enter to accept this name, or type a different name:")
        element_name = input("Element: ") or filename_base
        
        # Make sure we have a place to store file-specific settings
        if file_id not in input_data:
            input_data[file_id] = {}
        
        # Store element name in both places for backward compatibility
        input_data['element_name'] = element_name
        input_data[file_id]['element_name'] = element_name
    
    # Format element name for nice titles
    formatted_element = element_name
    # Try to identify if there's a number in the element name
    match = re.match(r'([a-zA-Z]+)(\d+)', element_name)
    if match:
        element, number = match.groups()
        # Capitalize the element and format with a hyphen
        formatted_element = f"{element.capitalize()}-{number}"
    
    try:
        # Read the data
        data = read_spectrum_data(filepath)
        
        print(f"\nSuccessfully loaded {formatted_element} data with {len(data)} channels")
        print(f"Max counts: {data['Counts'].max():.0f} at channel {data['Counts'].idxmax()}")
        
        # Store filepath in input_data
        if 'filepaths' not in input_data:
            input_data['filepaths'] = {}
        input_data['filepaths'][element_name] = filepath
        
        # Make sure we have a section for this file's config
        if file_id not in input_data:
            input_data[file_id] = {}
        
        # Set element name in file-specific config
        input_data[file_id]['element_name'] = element_name
        
        # Process the file with its file-specific config
        file_config = input_data[file_id]
        
        # Set output folder in file config
        if output_folder:
            file_config['output_folder'] = output_folder
            
        # Process multiple peaks for this single isotope
        isotope_data = process_multiple_peaks_one_isotope(data, element_name, output_folder, file_config)
        
        # Update the file-specific config with any changes from processing
        input_data[file_id] = file_config
        
        # Collect peak information for return
        all_peak_locations = isotope_data['peak_locations']
        all_peak_uncertainties = isotope_data['peak_uncertainties']
        peak_labels = []
        
        for j, (loc, err) in enumerate(zip(all_peak_locations, all_peak_uncertainties)):
            if len(all_peak_locations) > 1:
                peak_labels.append(f"{element_name}_peak{j+1}")
            else:
                peak_labels.append(element_name)
        
        # Display summary of peak locations and uncertainties
        if all_peak_locations:
            print("\n\n" + "="*70)
            print(f"SUMMARY OF {formatted_element} PEAK LOCATIONS AND UNCERTAINTIES".center(70))
            print("="*70)
            
            # Print labels for reference
            print("Peak Labels (for reference only):")
            for i, label in enumerate(peak_labels):
                print(f"{i+1}: {label}")
            
            print("\nPeak Locations (channels):")
            print(all_peak_locations)
            
            print("\nPeak Uncertainties (channels, 1σ):")
            print(all_peak_uncertainties)
            
            print("="*70)
            print("Copy-paste the above lists for your records!".center(70))
            print("="*70)
        
        # Save the configuration with all updates
        save_user_input(input_data, f"{element_name}_complete", output_folder)
        
        return {
            'element_name': element_name,
            'formatted_name': formatted_element,
            'results': isotope_data['results'],
            'peak_locations': all_peak_locations,
            'peak_uncertainties': all_peak_uncertainties,
            'peak_labels': peak_labels
        }
        
    except Exception as e:
        print(f"Error during analysis of {element_name}: {e}")
        return None

def process_multiple_isotopes(input_data={}):
    """
    Process multiple isotopes, each with potentially multiple peaks.
    
    Args:
        input_data: Optional pre-loaded input data
        
    Returns:
        Dictionary of isotopes, each containing a dictionary of peak IDs and their fit results
    """
    all_isotope_results = {}
    output_folder = None
    
    # For collecting all peak locations across all isotopes
    all_peak_locations = []
    all_peak_uncertainties = []
    all_peak_labels = []
    
    # Check if use_folder preference exists in input_data
    if 'use_folder' in input_data:
        use_folder = input_data['use_folder']
        print(f"\nUsing pre-loaded preference for folder selection: {'Yes' if use_folder else 'No'}")
    else:
        # Ask if user wants to specify a folder with CSV files
        use_folder = input("\nDo you want to specify a folder containing CSV files? (y/n): ").lower().startswith('y')
        # Save preference to input data
        input_data['use_folder'] = use_folder
    
    if use_folder:
        # Check if folder_path exists in input_data
        if 'folder_path' in input_data:
            folder_path = input_data['folder_path']
            print(f"\nUsing pre-loaded folder path: {folder_path}")
        else:
            # Get folder path
            folder_path = input("\nEnter the path to the folder containing CSV files: ")
            # Save folder path to input data
            input_data['folder_path'] = folder_path
        
        # Create output folder inside the input folder
        output_folder = ensure_output_folder(folder_path)
        
        # List all CSV files in the folder
        csv_files = list_csv_files(folder_path)
        
        if not csv_files:
            print("No CSV files found in the specified folder.")
            return {}
            
        print(f"\nFound {len(csv_files)} CSV files:")
        for i, file in enumerate(csv_files):
            print(f"{i+1}. {os.path.basename(file)}")
            
        # Check if file_selection is in input_data
        if 'file_selection' in input_data:
            file_selection = input_data['file_selection']
            print(f"\nUsing pre-loaded file selection: {file_selection}")
        else:
            # Ask which files to process manually
            file_selection = ask_for_file_selection()
            input_data['file_selection'] = file_selection
        
        if file_selection == 'all':
            selected_files = csv_files
        else:
            try:
                # Parse selection and convert to 0-based indices
                indices = [int(idx.strip()) - 1 for idx in file_selection.split(',')]
                selected_files = [csv_files[idx] for idx in indices if 0 <= idx < len(csv_files)]
                
                if not selected_files:
                    print("Warning: No valid files from saved selection. Please make a new selection.")
                    # Fall back to manual selection
                    file_selection = ask_for_file_selection()
                    input_data['file_selection'] = file_selection
                        
                    if file_selection == 'all':
                        selected_files = csv_files
                    else:
                        try:
                            indices = [int(idx.strip()) - 1 for idx in file_selection.split(',')]
                            selected_files = [csv_files[idx] for idx in indices if 0 <= idx < len(csv_files)]
                        except (ValueError, IndexError):
                            print("Invalid selection. No files will be processed.")
                            return {}
            except (ValueError, IndexError):
                print("Error in saved file selection. Please make a new selection.")
                # Fall back to manual selection
                file_selection = ask_for_file_selection()
                input_data['file_selection'] = file_selection
                
                if file_selection == 'all':
                    selected_files = csv_files
                else:
                    try:
                        indices = [int(idx.strip()) - 1 for idx in file_selection.split(',')]
                        selected_files = [csv_files[idx] for idx in indices if 0 <= idx < len(csv_files)]
                    except (ValueError, IndexError):
                        print("Invalid selection. No files will be processed.")
                        return {}
        
        # Process each selected file using the process_single_file function
        for i, filepath in enumerate(selected_files):
            print(f"\n\n=== Processing File {i+1} of {len(selected_files)}: {os.path.basename(filepath)} ===")
            
            # Process the file with a copy of the input_data to avoid cross-file configuration issues
            file_result = process_single_file(filepath, input_data, output_folder)
            
            if file_result:
                # Store results
                element_name = file_result['element_name']
                all_isotope_results[element_name] = file_result['results']
                
                # Collect peak information for summary display
                for j, (loc, err, label) in enumerate(zip(
                        file_result['peak_locations'], 
                        file_result['peak_uncertainties'],
                        file_result['peak_labels'])):
                    all_peak_locations.append(loc)
                    all_peak_uncertainties.append(err)
                    all_peak_labels.append(label)
    else:
        # Manual file entry mode
        # Create output folder in current directory
        output_folder = ensure_output_folder()
        
        # Ask how many isotopes to process
        while True:
            try:
                num_isotopes = int(input("\nHow many isotopes would you like to process? "))
                if num_isotopes <= 0:
                    print("Error: Please enter a positive number.")
                    continue
                break
            except ValueError:
                print("Error: Please enter a valid integer.")
        
        # Process each isotope
        for i in range(num_isotopes):
            print(f"\n\n=== Processing Isotope {i+1} of {num_isotopes} ===")
            
            # Get input filename from user
            print("Please enter the path to your spectrum data file:")
            filepath = input("Filename: ")
            
            # Check if file exists and handle relative paths
            if not os.path.isfile(filepath):
                # Try looking in the current directory
                if os.path.isfile(os.path.basename(filepath)):
                    filepath = os.path.basename(filepath)
                else:
                    print(f"Error: File '{filepath}' not found.")
                    continue
            
            # For manual entry, set element name in input_data here
            # This makes it clearer where the config for each file starts
            file_id = os.path.basename(filepath)
            
            # Get element name from user
            print("\nPlease enter the element name or isotope being analyzed (e.g., Co60, Na22):")
            element_name = input("Element: ") or os.path.splitext(file_id)[0]
            
            # Set up file-specific config
            if file_id not in input_data:
                input_data[file_id] = {'element_name': element_name}
            else:
                input_data[file_id]['element_name'] = element_name
            
            # Process the file using process_single_file
            file_result = process_single_file(filepath, input_data, output_folder)
            
            if file_result:
                # Store results
                element_name = file_result['element_name']
                all_isotope_results[element_name] = file_result['results']
                
                # Collect peak information for summary display
                for j, (loc, err, label) in enumerate(zip(
                        file_result['peak_locations'], 
                        file_result['peak_uncertainties'],
                        file_result['peak_labels'])):
                    all_peak_locations.append(loc)
                    all_peak_uncertainties.append(err)
                    all_peak_labels.append(label)
    
    # Display summary of all peak locations and uncertainties
    if all_peak_locations:
        print("\n\n" + "="*70)
        print("SUMMARY OF ALL PEAK LOCATIONS AND UNCERTAINTIES".center(70))
        print("="*70)
        
        # Print labels for reference (but not for copying)
        print("Peak Labels (for reference only):")
        for i, label in enumerate(all_peak_labels):
            print(f"{i+1}: {label}")
        
        print("\nPeak Locations (channels):")
        print(all_peak_locations)
        
        print("\nPeak Uncertainties (channels, 1σ):")
        print(all_peak_uncertainties)
        
        # Ask if user wants to convert to energy
        use_energy = input("\nDo you want to convert peak locations to energy (MeV)? (y/n): ").lower().startswith('y')
        
        if use_energy:
            # Check if energy conversion parameters already exist in input_data
            a = None
            b = None
            a_err = None
            b_err = None
            
            if 'energy_conversion' in input_data:
                energy_conversion = input_data['energy_conversion']
                a = energy_conversion.get('a')
                a_err = energy_conversion.get('a_err')
                b = energy_conversion.get('b')
                b_err = energy_conversion.get('b_err')
                
                if all(param is not None for param in [a, b, a_err, b_err]):
                    print(f"\nUsing pre-loaded energy conversion parameters:")
                    print(f"a = {a} ± {a_err}, b = {b} ± {b_err}")
            
            # If any parameters are missing, prompt for all of them
            if any(param is None for param in [a, b, a_err, b_err]):
                # Get conversion parameters (C = a + b*E)
                print("\nPlease enter the parameters for the conversion: Channel = a + b*Energy(MeV)")
                while True:
                    try:
                        a = float(input("Parameter a: "))
                        a_err = float(input("Uncertainty in a: "))
                        b = float(input("Parameter b: "))
                        b_err = float(input("Uncertainty in b: "))
                        
                        if b == 0:
                            print("Error: Parameter 'b' cannot be zero.")
                            continue
                        
                        break
                    except ValueError:
                        print("Error: Please enter valid numbers.")
            
            # Convert peak locations to energy with propagated errors
            energy_peaks = []
            energy_uncertainties = []
            
            for loc, err in zip(all_peak_locations, all_peak_uncertainties):
                # Convert channel to energy: E = (C-a)/b
                energy = (loc - a) / b
                
                # Propagate uncertainties
                # For function E = (C-a)/b, the uncertainty is:
                # σ_E^2 = (∂E/∂C)^2 * σ_C^2 + (∂E/∂a)^2 * σ_a^2 + (∂E/∂b)^2 * σ_b^2
                # where ∂E/∂C = 1/b, ∂E/∂a = -1/b, ∂E/∂b = -(C-a)/b^2
                
                dE_dC = 1/b
                dE_da = -1/b
                dE_db = -(loc-a)/(b**2)
                
                energy_err = np.sqrt((dE_dC**2 * err**2) + 
                                     (dE_da**2 * a_err**2) + 
                                     (dE_db**2 * b_err**2))
                
                energy_peaks.append(energy)
                energy_uncertainties.append(energy_err)
            
            # Store the energy conversion parameters in input_data
            if 'energy_conversion' not in input_data:
                input_data['energy_conversion'] = {
                    'a': a, 
                    'a_err': a_err,
                    'b': b,
                    'b_err': b_err
                }
            
            print("\n" + "="*70)
            print("ENERGY CONVERSION RESULTS".center(70))
            print("="*70)
            
            print("\nPeak Energies (MeV):")
            print([f"{e:.4f}" for e in energy_peaks])
            
            print("\nPeak Energy Uncertainties (MeV, 1σ):")
            print([f"{e:.4f}" for e in energy_uncertainties])
            
            print("\nDetailed Peak Energies:")
            for i, (energy, err, label) in enumerate(zip(energy_peaks, energy_uncertainties, all_peak_labels)):
                print(f"{i+1}: {label} = {energy:.4f} ± {err:.4f} MeV")
        
        print("\n" + "="*70)
        print("Copy-paste the above lists for your records!".center(70))
        print("="*70)
    
    # Save final complete configuration with all updates
    save_user_input(input_data, "all_isotopes_complete", output_folder)
    
    return all_isotope_results

def main():
    """Main function to run the spectrum analysis tool."""
    
    print("\n=== Radiation Spectrum Analysis Tool ===\n")
    print("This tool allows you to process multiple peaks from multiple isotopes.")
    
    # Ask if user wants to use a saved input file
    use_saved_input = input("\nDo you want to use a saved input file? (y/n): ").lower().startswith('y')
    
    # Initialize input_data as an empty dictionary by default
    input_data = {}
    
    if use_saved_input:
        # Get input file path
        input_filepath = input("\nEnter the path to the input file: ")
        
        # Check if file exists
        if not os.path.isfile(input_filepath):
            print(f"Error: File '{input_filepath}' not found.")
        else:
            # Load input data
            loaded_data = load_user_input_from_file(input_filepath)
            if loaded_data:
                input_data = loaded_data
            else:
                print("Error loading input file. Continuing with manual input.")
    
    # Check if processing mode is already configured
    if 'processing_mode' in input_data:
        processing_mode = input_data['processing_mode']
        is_single_file = processing_mode == 'single_file'
        print(f"\nUsing pre-configured processing mode: {'single file' if is_single_file else 'multiple files'}")
        choice = '1' if is_single_file else '2'
    else:
        # Ask if user wants to process multiple files or a single file
        while True:
            choice = input("\nDo you want to process (1) a single file or (2) multiple files? (1/2): ")
            if choice in ['1', '2']:
                break
            print("Error: Please enter 1 or 2.")
        
        # Store choice in input_data
        input_data['processing_mode'] = 'single_file' if choice == '1' else 'multiple_files'
    
    if choice == '1':
        # Single file processing
        
        # Create output folder in current directory
        output_folder = ensure_output_folder()
        
        # Check if use_folder preference exists in input_data
        if 'use_folder' in input_data:
            use_folder = input_data['use_folder']
            print(f"\nUsing pre-loaded preference for folder selection: {'Yes' if use_folder else 'No'}")
        else:
            # Ask if user wants to select from a folder
            use_folder = input("\nDo you want to select from a folder containing CSV files? (y/n): ").lower().startswith('y')
            # Save preference to input data
            input_data['use_folder'] = use_folder
        
        filepath = ""
        if use_folder:
            # Check if folder_path exists in input_data
            if 'folder_path' in input_data:
                folder_path = input_data['folder_path']
                print(f"\nUsing pre-loaded folder path: {folder_path}")
            else:
                # Get folder path
                folder_path = input("\nEnter the path to the folder containing CSV files: ")
                # Save folder path to input data
                input_data['folder_path'] = folder_path
            
            # Create output folder inside input folder
            output_folder = ensure_output_folder(folder_path)
            
            # List all CSV files in the folder
            csv_files = list_csv_files(folder_path)
            
            if not csv_files:
                print("No CSV files found in the specified folder.")
                return None
            
            # Display the files
            print(f"\nFound {len(csv_files)} CSV files:")
            for i, file in enumerate(csv_files):
                print(f"{i+1}. {os.path.basename(file)}")
            
            # Check if file_selection exists in input_data
            if 'file_selection' in input_data:
                file_idx = input_data['file_selection']
                print(f"\nUsing pre-loaded file selection: {file_idx + 1}")
                if 0 <= file_idx < len(csv_files):
                    filepath = csv_files[file_idx]
                else:
                    print(f"Error: Saved file index {file_idx + 1} is out of range. Files available: {len(csv_files)}")
                    # Fall back to manual selection
                    while True:
                        try:
                            file_idx = int(input("\nEnter the number of the file to process: ")) - 1
                            if 0 <= file_idx < len(csv_files):
                                filepath = csv_files[file_idx]
                                # Save the selection
                                input_data['file_selection'] = file_idx
                                break
                            else:
                                print(f"Error: Please enter a number between 1 and {len(csv_files)}.")
                        except ValueError:
                            print("Error: Please enter a valid number.")
            else:
                # Ask which file to process
                while True:
                    try:
                        file_idx = int(input("\nEnter the number of the file to process: ")) - 1
                        if 0 <= file_idx < len(csv_files):
                            filepath = csv_files[file_idx]
                            # Save the selection
                            input_data['file_selection'] = file_idx
                            break
                        else:
                            print(f"Error: Please enter a number between 1 and {len(csv_files)}.")
                    except ValueError:
                        print("Error: Please enter a valid number.")
        else:
            # Check if direct_filepath is in input_data
            if 'direct_filepath' in input_data:
                filepath = input_data['direct_filepath']
                print(f"\nUsing pre-loaded file path: {filepath}")
                if not os.path.isfile(filepath):
                    print(f"Error: Saved file path '{filepath}' not found.")
                    # Fall back to manual entry
                    print("Please enter the path to your spectrum data file:")
                    filepath = input("Filename: ")
                    
                    # Check and resolve the path
                    if not os.path.isfile(filepath):
                        if os.path.isfile(os.path.basename(filepath)):
                            filepath = os.path.basename(filepath)
                        else:
                            print(f"Error: File '{filepath}' not found.")
                            return None
                            
                    # Save the new path
                    input_data['direct_filepath'] = filepath
            else:
                # Get input filename from user directly
                print("Please enter the path to your spectrum data file:")
                filepath = input("Filename: ")
                
                # Check if file exists and handle relative paths
                if not os.path.isfile(filepath):
                    # Try looking in the current directory
                    if os.path.isfile(os.path.basename(filepath)):
                        filepath = os.path.basename(filepath)
                    else:
                        print(f"Error: File '{filepath}' not found.")
                        return None
                        
                # Save the filepath
                input_data['direct_filepath'] = filepath
        
        # Process single file with our core function
        return process_single_file(filepath, input_data, output_folder)
    else:
        # Multiple files processing
        return process_multiple_isotopes(input_data)

if __name__ == "__main__":
    main()