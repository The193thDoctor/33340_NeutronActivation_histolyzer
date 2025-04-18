import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
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
        for channel, label in peak_labels:
            plt.annotate(label,
                         xy=(channel, data.loc[data['Channel'] == channel, 'Counts'].values[0] 
                             if channel in data['Channel'].values 
                             else data['Counts'].max() * 0.9),
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
                   alpha=0.2, color='red', label='1σ Confidence')
        plt.axvline(x=mu, color='k', linestyle='--')
        plt.xlabel('Channel')
        plt.ylabel('Counts (Background Subtracted)')
        plt.title(f'{peak_title} Peak with Gaussian Fit & Uncertainty')
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
        return {
            'element': element_name,
            'peak_id': peak_id,
            'center': max_channel,
            'center_err': (end_ch - start_ch) / 10,  # Conservative fallback
            'amplitude': max_subtracted,
            'amplitude_err': np.sqrt(max_subtracted) * 2  # Poisson error × 2
        }

def process_peak(data, element_name, peak_id=None, output_folder=None):
    """
    Process a single peak with user input for channel selection.
    
    Args:
        data: DataFrame with channels and counts
        element_name: Name of the element being analyzed
        peak_id: Optional identifier for distinguishing multiple peaks
        output_folder: Optional output folder path
    
    Returns:
        Fit results for this peak
    """
    # Create a formatted title string based on peak_id
    peak_title = f"{element_name}"
    if peak_id:
        peak_title = f"{element_name} Peak {peak_id}"
    
    print(f"\n=== Processing {peak_title} ===")
    
    # Query user for two channel locations
    print("\nYou need to select two channels to define the background region.")
    while True:
        try:
            start_ch = int(input(f"Enter starting channel for {peak_title}: "))
            if start_ch not in data['Channel'].values:
                print(f"Warning: Channel {start_ch} not found in data. Available range: "
                      f"{data['Channel'].min()} to {data['Channel'].max()}")
                continue
            
            end_ch = int(input(f"Enter ending chclaudeannel for {peak_title}: "))
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

def process_multiple_peaks_one_isotope(data, element_name, output_folder=None):
    """
    Process multiple peaks for a single isotope.
    
    Args:
        data: DataFrame with channels and counts
        element_name: Name of the element being analyzed
        output_folder: Optional output folder path
    
    Returns:
        Dictionary containing peak results and additional information
    """
    # Plot the initial spectrum without peak labels
    title = f"{element_name} Radiation Spectrum"
    plot_histogram(data, element_name, title=title, output_folder=output_folder)
    
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
    
    # Process each peak
    all_results = {}
    peak_locations = []
    peak_uncertainties = []
    
    for i in range(num_peaks):
        # For a single peak, use None as peak_id. For multiple peaks, use numbers.
        peak_id = f"{i+1}" if num_peaks > 1 else None
        fit_results = process_peak(data, element_name, peak_id, output_folder)
        all_results[peak_id or 'main'] = fit_results
        
        # Store peak locations and uncertainties, converting numpy values to Python float
        peak_locations.append(float(fit_results['center']))
        peak_uncertainties.append(float(fit_results['center_err']))
    
    # Ask if user wants to create a labeled spectrum with the fitted peak positions
    if peak_locations:
        label_peaks = input("\nDo you want to create a spectrum with labeled peak positions? (y/n): ").lower().startswith('y')
        
        if label_peaks:
            # Ask if user wants to convert to energy (MeV)
            use_energy = input("\nDo you want to convert the x-axis to energy (MeV)? (y/n): ").lower().startswith('y')
            
            a = None
            b = None
            a_err = None
            b_err = None
            
            if use_energy:
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
                for channel, label in peak_labels:
                    energy_value = (channel - a) / b
                    plt.annotate(label,
                                 xy=(energy_value, data.loc[data['Channel'] == channel, 'Counts'].values[0] 
                                     if channel in data['Channel'].values 
                                     else data['Counts'].max() * 0.9),
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
                for i, (loc, err) in enumerate(zip(peak_locations, peak_uncertainties)):
                    peak_label = f"Ch: {loc:.1f}±{err:.1f}"
                    # Find the closest channel in the data
                    closest_channel = int(round(loc))
                    peak_labels.append((closest_channel, peak_label))
                
                # Create a new plot with the fitted peak positions labeled
                title = f"{element_name} Radiation Spectrum (Fitted Peaks)"
                file_prefix = f"{element_name}_labeled"
                plot_histogram(data, element_name, title=title, file_prefix=file_prefix, 
                              output_folder=output_folder, peak_labels=peak_labels)
    
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

def process_multiple_isotopes():
    """
    Process multiple isotopes, each with potentially multiple peaks.
    
    Returns:
        Dictionary of isotopes, each containing a dictionary of peak IDs and their fit results
    """
    all_isotope_results = {}
    output_folder = None
    
    # For collecting all peak locations across all isotopes
    all_peak_locations = []
    all_peak_uncertainties = []
    peak_labels = []
    
    # Ask if user wants to specify a folder with CSV files
    use_folder = input("\nDo you want to specify a folder containing CSV files? (y/n): ").lower().startswith('y')
    
    if use_folder:
        # Get folder path
        folder_path = input("\nEnter the path to the folder containing CSV files: ")
        
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
            
        # Ask which files to process
        print("\nWhich files would you like to process?")
        print("Enter file numbers separated by commas (e.g., 1,3,5), or 'all' for all files:")
        file_selection = input("Selection: ").strip().lower()
        
        if file_selection == 'all':
            selected_files = csv_files
        else:
            try:
                # Parse selection and convert to 0-based indices
                indices = [int(idx.strip()) - 1 for idx in file_selection.split(',')]
                selected_files = [csv_files[idx] for idx in indices if 0 <= idx < len(csv_files)]
            except (ValueError, IndexError):
                print("Invalid selection. No files will be processed.")
                return {}
        
        # Process each selected file
        for i, filepath in enumerate(selected_files):
            print(f"\n\n=== Processing File {i+1} of {len(selected_files)}: {os.path.basename(filepath)} ===")
            
            # Get element name from user or use filename
            filename_base = os.path.splitext(os.path.basename(filepath))[0]
            print(f"\nSuggested element name based on filename: {filename_base}")
            print("Press Enter to accept this name, or type a different name:")
            element_name = input("Element: ") or filename_base
            
            # Format element name for nice titles
            # This handles cases like "co60" -> "Co-60" or "na22" -> "Na-22"
            formatted_element = element_name
            # Try to identify if there's a number in the element name
            import re
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
                
                # Process multiple peaks for this isotope
                isotope_data = process_multiple_peaks_one_isotope(data, element_name, output_folder)
                
                # Store results
                all_isotope_results[element_name] = isotope_data['results']
                
                # Collect peak information
                for j, (loc, err) in enumerate(zip(isotope_data['peak_locations'], isotope_data['peak_uncertainties'])):
                    all_peak_locations.append(loc)
                    all_peak_uncertainties.append(err)
                    if len(isotope_data['peak_locations']) > 1:
                        # Use consistent format with the numbered peak_id
                        peak_labels.append(f"{element_name} Peak {j+1}")
                    else:
                        peak_labels.append(element_name)
                
            except Exception as e:
                print(f"Error during analysis of {element_name}: {e}")
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
            
            # Get element name from user
            print("\nPlease enter the element name or isotope being analyzed (e.g., Co60, Na22):")
            element_name = input("Element: ")
            if not element_name:
                # Use filename as fallback
                element_name = os.path.splitext(os.path.basename(filepath))[0]
            
            # Format element name for nice titles
            # This handles cases like "co60" -> "Co-60" or "na22" -> "Na-22"
            formatted_element = element_name
            # Try to identify if there's a number in the element name
            import re
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
                
                # Process multiple peaks for this isotope
                isotope_data = process_multiple_peaks_one_isotope(data, element_name, output_folder)
                
                # Store results
                all_isotope_results[element_name] = isotope_data['results']
                
                # Collect peak information
                for j, (loc, err) in enumerate(zip(isotope_data['peak_locations'], isotope_data['peak_uncertainties'])):
                    all_peak_locations.append(loc)
                    all_peak_uncertainties.append(err)
                    if len(isotope_data['peak_locations']) > 1:
                        # Use consistent format with the numbered peak_id
                        peak_labels.append(f"{element_name} Peak {j+1}")
                    else:
                        peak_labels.append(element_name)
                
            except Exception as e:
                print(f"Error during analysis of {element_name}: {e}")
    
    # Display summary of all peak locations and uncertainties
    if all_peak_locations:
        print("\n\n" + "="*70)
        print("SUMMARY OF ALL PEAK LOCATIONS AND UNCERTAINTIES".center(70))
        print("="*70)
        
        # Print labels for reference (but not for copying)
        print("Peak Labels (for reference only):")
        for i, label in enumerate(peak_labels):
            print(f"{i+1}: {label}")
        
        print("\nPeak Locations (channels):")
        print(all_peak_locations)
        
        print("\nPeak Uncertainties (channels, 1σ):")
        print(all_peak_uncertainties)
        
        # Ask if user wants to convert to energy
        use_energy = input("\nDo you want to convert peak locations to energy (MeV)? (y/n): ").lower().startswith('y')
        
        if use_energy:
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
            
            print("\n" + "="*70)
            print("ENERGY CONVERSION RESULTS".center(70))
            print("="*70)
            
            print("\nPeak Energies (MeV):")
            print([f"{e:.4f}" for e in energy_peaks])
            
            print("\nPeak Energy Uncertainties (MeV, 1σ):")
            print([f"{e:.4f}" for e in energy_uncertainties])
            
            print("\nDetailed Peak Energies:")
            for i, (energy, err, label) in enumerate(zip(energy_peaks, energy_uncertainties, peak_labels)):
                print(f"{i+1}: {label} = {energy:.4f} ± {err:.4f} MeV")
        
        print("\n" + "="*70)
        print("Copy-paste the above lists for your records!".center(70))
        print("="*70)
    
    return all_isotope_results

def main():
    """Main function to run the spectrum analysis tool."""
    
    print("\n=== Radiation Spectrum Analysis Tool ===\n")
    print("This tool allows you to process multiple peaks from multiple isotopes.")
    
    # Ask if user wants to process multiple isotopes or a single file
    while True:
        choice = input("\nDo you want to process (1) a single file or (2) multiple isotopes? (1/2): ")
        if choice in ['1', '2']:
            break
        print("Error: Please enter 1 or 2.")
    
    if choice == '1':
        # Create output folder in current directory
        output_folder = ensure_output_folder()
        
        # For collecting peak information
        all_peak_locations = []
        all_peak_uncertainties = []
        peak_labels = []
        
        # Ask if user wants to select from a folder
        use_folder = input("\nDo you want to select from a folder containing CSV files? (y/n): ").lower().startswith('y')
        
        filepath = ""
        if use_folder:
            # Get folder path
            folder_path = input("\nEnter the path to the folder containing CSV files: ")
            
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
            
            # Ask which file to process
            while True:
                try:
                    file_idx = int(input("\nEnter the number of the file to process: ")) - 1
                    if 0 <= file_idx < len(csv_files):
                        filepath = csv_files[file_idx]
                        break
                    else:
                        print(f"Error: Please enter a number between 1 and {len(csv_files)}.")
                except ValueError:
                    print("Error: Please enter a valid number.")
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
        
        # Get element name from user or use filename
        filename_base = os.path.splitext(os.path.basename(filepath))[0]
        print(f"\nSuggested element name based on filename: {filename_base}")
        print("Press Enter to accept this name, or type a different name:")
        element_name = input("Element: ") or filename_base
        
        # Format element name for nice titles
        # This handles cases like "co60" -> "Co-60" or "na22" -> "Na-22"
        formatted_element = element_name
        # Try to identify if there's a number in the element name
        import re
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
            
            # Process multiple peaks for this single isotope
            isotope_data = process_multiple_peaks_one_isotope(data, element_name, output_folder)
            
            # Collect peak information
            for j, (loc, err) in enumerate(zip(isotope_data['peak_locations'], isotope_data['peak_uncertainties'])):
                all_peak_locations.append(loc)
                all_peak_uncertainties.append(err)
                if len(isotope_data['peak_locations']) > 1:
                    peak_labels.append(f"{element_name}_peak{j+1}")
                else:
                    peak_labels.append(element_name)
            
            # Display summary of all peak locations and uncertainties
            if all_peak_locations:
                print("\n\n" + "="*70)
                print("SUMMARY OF ALL PEAK LOCATIONS AND UNCERTAINTIES".center(70))
                print("="*70)
                
                # Print labels for reference (but not for copying)
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
            
            return {element_name: isotope_data['results']}
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            return None
    else:
        # Process multiple isotopes
        return process_multiple_isotopes()

if __name__ == "__main__":
    main()