import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from ase.io import read, write
from ase import Atoms
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import warnings
warnings.filterwarnings("ignore")

class RelaxationComparator:
    """
    Compare relaxed crystal structures from different methods (M3GNet vs DeepRelax).
    """
    
    def __init__(self, m3gnet_dir, deeprelax_dir, original_dir=None):
        """
        Initialize comparator with directories containing relaxed structures.
        
        Parameters:
        -----------
        m3gnet_dir : str
            Directory with M3GNet relaxed CIF files
        deeprelax_dir : str
            Directory with DeepRelax relaxed CIF files  
        original_dir : str, optional
            Directory with original (unrelaxed) structures for reference
        """
        self.m3gnet_dir = Path(m3gnet_dir)
        self.deeprelax_dir = Path(deeprelax_dir)
        self.original_dir = Path(original_dir) if original_dir else None
        
        self.results = []
    
    def find_matching_pairs(self):
        """
        Find matching CIF files between M3GNet and DeepRelax directories.
        
        Returns:
        --------
        list
            List of tuples (structure_id, m3gnet_file, deeprelax_file, original_file)
        """
        # Get all CIF files from both directories
        m3gnet_files = {f.stem: f for f in self.m3gnet_dir.glob("*.cif")}
        deeprelax_files = {f.stem: f for f in self.deeprelax_dir.glob("*.cif")}
        
        # Find common structure IDs
        common_ids = set(m3gnet_files.keys()) & set(deeprelax_files.keys())
        
        pairs = []
        for structure_id in common_ids:
            m3gnet_file = m3gnet_files[structure_id]
            deeprelax_file = deeprelax_files[structure_id]
            
            # Find original file if directory provided
            original_file = None
            if self.original_dir:
                possible_original = self.original_dir / f"{structure_id}.cif"
                if possible_original.exists():
                    original_file = possible_original
                else:
                    # Try alternative naming conventions
                    for pattern in [f"{structure_id}_unrelaxed.cif", f"{structure_id}_original.cif"]:
                        alt_original = self.original_dir / pattern
                        if alt_original.exists():
                            original_file = alt_original
                            break
            
            pairs.append((structure_id, m3gnet_file, deeprelax_file, original_file))
        
        print(f"Found {len(pairs)} matching structure pairs")
        return pairs
    
    def find_alternative_matches(self):
        """
        Try alternative file matching strategies when exact matching fails.
        """
        m3gnet_files = {f.name: f for f in self.m3gnet_dir.glob("*.cif")}
        deeprelax_files = {f.name: f for f in self.deeprelax_dir.glob("*.cif")}
        
        pairs = []
        
        # Strategy 1: Remove common suffixes and try matching
        print("   Trying suffix removal matching...")
        suffixes_to_remove = ['_relaxed', '_predicted', '_m3gnet', '_deeprelax', '_final']
        
        for suffix in suffixes_to_remove:
            m3g_cleaned = {}
            deep_cleaned = {}
            
            # Clean M3GNet filenames
            for name, path in m3gnet_files.items():
                clean_name = name.replace(suffix, '').replace('.cif', '')
                if clean_name:
                    m3g_cleaned[clean_name] = path
            
            # Clean DeepRelax filenames  
            for name, path in deeprelax_files.items():
                clean_name = name.replace(suffix, '').replace('.cif', '')
                if clean_name:
                    deep_cleaned[clean_name] = path
            
            # Find matches
            matches = set(m3g_cleaned.keys()) & set(deep_cleaned.keys())
            if matches:
                for match in matches:
                    pairs.append((match, m3g_cleaned[match], deep_cleaned[match], None))
                print(f"   Found {len(matches)} matches by removing '{suffix}'")
                break
        
        # Strategy 2: Pattern replacement matching
        if not pairs:
            print("   Trying pattern replacement matching...")
            patterns = [
                ('_relaxed', '_predicted'),
                ('_m3gnet', '_deeprelax'),
                ('relaxed_', 'predicted_'),
                ('m3gnet_', 'deeprelax_'),
            ]
            
            for m3g_pattern, deep_pattern in patterns:
                temp_pairs = []
                for m3g_name, m3g_path in m3gnet_files.items():
                    if m3g_pattern in m3g_name:
                        expected_deep_name = m3g_name.replace(m3g_pattern, deep_pattern)
                        if expected_deep_name in deeprelax_files:
                            structure_id = m3g_name.replace(m3g_pattern, '').replace('.cif', '')
                            temp_pairs.append((structure_id, m3g_path, deeprelax_files[expected_deep_name], None))
                
                if temp_pairs:
                    pairs.extend(temp_pairs)
                    print(f"   Found {len(temp_pairs)} matches with pattern '{m3g_pattern}' → '{deep_pattern}'")
                    break
        
        # Strategy 3: Fuzzy matching by numbers
        if not pairs:
            print("   Trying numeric ID matching...")
            import re
            
            m3g_numbers = {}
            deep_numbers = {}
            
            for name, path in m3gnet_files.items():
                numbers = re.findall(r'\d+', name)
                if numbers:
                    key = numbers[0]  # Use first number found
                    if key not in m3g_numbers:
                        m3g_numbers[key] = []
                    m3g_numbers[key].append((name, path))
            
            for name, path in deeprelax_files.items():
                numbers = re.findall(r'\d+', name)
                if numbers:
                    key = numbers[0]  # Use first number found
                    if key not in deep_numbers:
                        deep_numbers[key] = []
                    deep_numbers[key].append((name, path))
            
            # Match by numbers
            for number in m3g_numbers:
                if (number in deep_numbers and 
                    len(m3g_numbers[number]) == 1 and 
                    len(deep_numbers[number]) == 1):
                    
                    m3g_name, m3g_path = m3g_numbers[number][0]
                    deep_name, deep_path = deep_numbers[number][0]
                    pairs.append((f"struct_{number}", m3g_path, deep_path, None))
            
            if pairs:
                print(f"   Found {len(pairs)} matches by numeric ID")
        
        return pairs
    
    def align_structures(self, atoms1, atoms2):
        """
        Align two structures by finding optimal atom correspondence.
        
        Parameters:
        -----------
        atoms1, atoms2 : ase.Atoms
            Structures to align
            
        Returns:
        --------
        tuple
            (aligned_atoms2, reorder_indices)
        """
        if len(atoms1) != len(atoms2):
            raise ValueError("Structures have different numbers of atoms")
        
        # Check if same composition
        if atoms1.get_chemical_formula() != atoms2.get_chemical_formula():
            print(f"Warning: Different compositions - {atoms1.get_chemical_formula()} vs {atoms2.get_chemical_formula()}")
        
        # For structures with same atom ordering, no alignment needed
        if np.array_equal(atoms1.get_atomic_numbers(), atoms2.get_atomic_numbers()):
            return atoms2, np.arange(len(atoms2))
        
        # Find optimal atom correspondence using Hungarian algorithm
        pos1 = atoms1.get_positions()
        pos2 = atoms2.get_positions()
        atomic_nums1 = atoms1.get_atomic_numbers()
        atomic_nums2 = atoms2.get_atomic_numbers()
        
        # Calculate distance matrix, with penalty for different elements
        distance_matrix = cdist(pos1, pos2)
        
        # Add large penalty for different atomic numbers
        for i in range(len(atoms1)):
            for j in range(len(atoms2)):
                if atomic_nums1[i] != atomic_nums2[j]:
                    distance_matrix[i, j] += 1000  # Large penalty
        
        # Solve assignment problem
        row_indices, col_indices = linear_sum_assignment(distance_matrix)
        
        # Reorder atoms2 to match atoms1
        aligned_atoms2 = atoms2[col_indices]
        
        return aligned_atoms2, col_indices
    
    def calculate_rmsd(self, atoms1, atoms2):
        """
        Calculate Root Mean Square Deviation between two structures.
        """
        try:
            aligned_atoms2, _ = self.align_structures(atoms1, atoms2)
            pos1 = atoms1.get_positions()
            pos2 = aligned_atoms2.get_positions()
            
            # Calculate RMSD
            diff = pos1 - pos2
            rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
            return rmsd
        except Exception as e:
            print(f"RMSD calculation failed: {e}")
            return np.nan
    
    def calculate_cell_differences(self, atoms1, atoms2):
        """
        Calculate differences in cell parameters.
        """
        cell1 = atoms1.get_cell()
        cell2 = atoms2.get_cell()
        
        # Cell lengths
        lengths1 = cell1.lengths()
        lengths2 = cell2.lengths()
        length_diff = np.abs(lengths1 - lengths2)
        
        # Cell angles  
        angles1 = cell1.angles()
        angles2 = cell2.angles()
        angle_diff = np.abs(angles1 - angles2)
        
        # Volume difference
        vol1 = atoms1.get_volume()
        vol2 = atoms2.get_volume()
        vol_diff = abs(vol1 - vol2)
        vol_percent = 100 * vol_diff / vol1 if vol1 > 0 else 0
        
        return {
            'length_diff_a': length_diff[0],
            'length_diff_b': length_diff[1], 
            'length_diff_c': length_diff[2],
            'angle_diff_alpha': angle_diff[0],
            'angle_diff_beta': angle_diff[1],
            'angle_diff_gamma': angle_diff[2],
            'volume_diff': vol_diff,
            'volume_percent_diff': vol_percent,
            'volume_m3gnet': vol1,
            'volume_deeprelax': vol2
        }
    
    def calculate_bond_differences(self, atoms1, atoms2, cutoff=3.0):
        """
        Calculate differences in nearest neighbor bond lengths.
        """
        try:
            aligned_atoms2, _ = self.align_structures(atoms1, atoms2)
            
            from ase.neighborlist import NeighborList
            
            # Build neighbor lists
            cutoffs = [cutoff/2] * len(atoms1)
            nl1 = NeighborList(cutoffs, self_interaction=False, bothways=True)
            nl2 = NeighborList(cutoffs, self_interaction=False, bothways=True)
            
            nl1.update(atoms1)
            nl2.update(aligned_atoms2)
            
            bonds1 = []
            bonds2 = []
            
            for i in range(len(atoms1)):
                indices1, offsets1 = nl1.get_neighbors(i)
                indices2, offsets2 = nl2.get_neighbors(i)
                
                if len(indices1) != len(indices2):
                    continue  # Different coordination, skip
                
                for j, offset in zip(indices1, offsets1):
                    if i < j:  # Avoid double counting
                        pos1 = atoms1.get_positions()
                        pos2 = aligned_atoms2.get_positions()
                        cell1 = atoms1.get_cell()
                        cell2 = aligned_atoms2.get_cell()
                        
                        bond1 = np.linalg.norm(pos1[j] + offset @ cell1 - pos1[i])
                        bond2 = np.linalg.norm(pos2[j] + offset @ cell2 - pos2[i])
                        
                        bonds1.append(bond1)
                        bonds2.append(bond2)
            
            if bonds1 and bonds2:
                bond_diff = np.mean(np.abs(np.array(bonds1) - np.array(bonds2)))
                return bond_diff
            else:
                return np.nan
                
        except Exception as e:
            print(f"Bond calculation failed: {e}")
            return np.nan
    
    def compare_single_pair(self, structure_id, m3gnet_file, deeprelax_file, original_file=None):
        """
        Compare a single pair of relaxed structures.
        """
        try:
            # Load structures
            m3gnet_atoms = read(str(m3gnet_file))
            deeprelax_atoms = read(str(deeprelax_file))
            original_atoms = read(str(original_file)) if original_file else None
            
            # Basic properties
            result = {
                'structure_id': structure_id,
                'n_atoms': len(m3gnet_atoms),
                'formula': m3gnet_atoms.get_chemical_formula(),
            }
            
            # RMSD between M3GNet and DeepRelax
            result['rmsd_m3gnet_deeprelax'] = self.calculate_rmsd(m3gnet_atoms, deeprelax_atoms)
            
            # Cell parameter differences
            cell_diff = self.calculate_cell_differences(m3gnet_atoms, deeprelax_atoms)
            result.update(cell_diff)
            
            # Bond length differences
            result['bond_diff'] = self.calculate_bond_differences(m3gnet_atoms, deeprelax_atoms)
            
            # If original structure available, calculate relaxation distances
            if original_atoms:
                result['rmsd_original_m3gnet'] = self.calculate_rmsd(original_atoms, m3gnet_atoms)
                result['rmsd_original_deeprelax'] = self.calculate_rmsd(original_atoms, deeprelax_atoms)
                
                # Volume changes from original
                vol_orig = original_atoms.get_volume()
                vol_m3g = m3gnet_atoms.get_volume()
                vol_deep = deeprelax_atoms.get_volume()
                
                result['volume_change_m3gnet'] = 100 * (vol_m3g - vol_orig) / vol_orig
                result['volume_change_deeprelax'] = 100 * (vol_deep - vol_orig) / vol_orig
            
            return result
            
        except Exception as e:
            print(f"Error comparing {structure_id}: {e}")
            return {
                'structure_id': structure_id,
                'error': str(e)
            }
    
    def compare_all(self):
        """
        Compare all matching pairs and return results DataFrame.
        """
        pairs = self.find_matching_pairs()
        
        if len(pairs) == 0:
            print("❌ No matching files found!")
            print("Debugging directory contents...")
            
            # Debug what files exist
            m3gnet_files = list(self.m3gnet_dir.glob("*.cif"))
            deeprelax_files = list(self.deeprelax_dir.glob("*.cif"))
            
            print(f"\nM3GNet directory ({self.m3gnet_dir}):")
            print(f"  Found {len(m3gnet_files)} CIF files")
            if m3gnet_files:
                print("  Sample files:")
                for f in m3gnet_files[:5]:
                    print(f"    - {f.name}")
            
            print(f"\nDeepRelax directory ({self.deeprelax_dir}):")
            print(f"  Found {len(deeprelax_files)} CIF files")
            if deeprelax_files:
                print("  Sample files:")
                for f in deeprelax_files[:5]:
                    print(f"    - {f.name}")
            
            # Try alternative matching strategies
            print("\n🔍 Trying alternative matching strategies...")
            alternative_pairs = self.find_alternative_matches()
            
            if alternative_pairs:
                print(f"Found {len(alternative_pairs)} alternative matches!")
                pairs = alternative_pairs
            else:
                print("No alternative matches found either.")
                # Return empty DataFrame with expected columns
                return pd.DataFrame(columns=['structure_id', 'rmsd_m3gnet_deeprelax', 'error'])
        
        print("Comparing relaxed structures...")
        results = []
        
        for structure_id, m3gnet_file, deeprelax_file, original_file in pairs:
            print(f"Processing {structure_id}...")
            result = self.compare_single_pair(structure_id, m3gnet_file, deeprelax_file, original_file)
            results.append(result)
        
        self.results = results
        df = pd.DataFrame(results)
        
        # Check if we have the expected columns before trying to filter
        if 'rmsd_m3gnet_deeprelax' in df.columns:
            valid_df = df.dropna(subset=['rmsd_m3gnet_deeprelax'])
            print(f"\nComparison complete!")
            print(f"Successfully compared: {len(valid_df)} structures")
            print(f"Failed comparisons: {len(df) - len(valid_df)}")
        else:
            print(f"\nComparison attempted but no successful RMSD calculations")
            print(f"Total attempts: {len(df)}")
            # Show what columns we do have
            if len(df) > 0:
                print(f"Available columns: {list(df.columns)}")
        
        return df
    
    def plot_comparison_summary(self, df, save_path=None):
        """
        Create summary plots comparing M3GNet and DeepRelax relaxations.
        """
        # Check if we have the required column
        if 'rmsd_m3gnet_deeprelax' not in df.columns:
            print("❌ No RMSD data available for plotting")
            print(f"Available columns: {list(df.columns)}")
            return
        
        # Filter valid results
        valid_df = df.dropna(subset=['rmsd_m3gnet_deeprelax'])
        
        if len(valid_df) == 0:
            print("❌ No valid data for plotting")
            return
        
        print(f"📊 Creating plots for {len(valid_df)} valid comparisons...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('M3GNet vs DeepRelax Relaxation Comparison', fontsize=16, fontweight='bold')
        
        # 1. RMSD distribution
        axes[0, 0].hist(valid_df['rmsd_m3gnet_deeprelax'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_xlabel('RMSD between M3GNet and DeepRelax (Å)')
        axes[0, 0].set_ylabel('Number of Structures')
        axes[0, 0].set_title('Position Differences (RMSD)')
        axes[0, 0].axvline(valid_df['rmsd_m3gnet_deeprelax'].median(), color='red', linestyle='--', 
                          label=f'Median: {valid_df["rmsd_m3gnet_deeprelax"].median():.3f} Å')
        axes[0, 0].legend()
        
        # 2. Volume comparison
        if 'volume_m3gnet' in valid_df.columns and 'volume_deeprelax' in valid_df.columns:
            axes[0, 1].scatter(valid_df['volume_m3gnet'], valid_df['volume_deeprelax'], alpha=0.6)
            min_vol = min(valid_df['volume_m3gnet'].min(), valid_df['volume_deeprelax'].min())
            max_vol = max(valid_df['volume_m3gnet'].max(), valid_df['volume_deeprelax'].max())
            axes[0, 1].plot([min_vol, max_vol], [min_vol, max_vol], 'r--', label='Perfect Agreement')
            axes[0, 1].set_xlabel('M3GNet Volume (Å³)')
            axes[0, 1].set_ylabel('DeepRelax Volume (Å³)')
            axes[0, 1].set_title('Volume Comparison')
            axes[0, 1].legend()
        
        # 3. Volume difference distribution
        if 'volume_percent_diff' in valid_df.columns:
            axes[0, 2].hist(valid_df['volume_percent_diff'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
            axes[0, 2].set_xlabel('Volume Difference (%)')
            axes[0, 2].set_ylabel('Number of Structures')
            axes[0, 2].set_title('Volume Differences')
            axes[0, 2].axvline(valid_df['volume_percent_diff'].median(), color='red', linestyle='--',
                              label=f'Median: {valid_df["volume_percent_diff"].median():.2f}%')
            axes[0, 2].legend()
        
        # 4. Cell parameter differences
        cell_params = ['length_diff_a', 'length_diff_b', 'length_diff_c']
        if all(param in valid_df.columns for param in cell_params):
            cell_data = valid_df[cell_params].values.flatten()
            axes[1, 0].hist(cell_data, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
            axes[1, 0].set_xlabel('Cell Length Difference (Å)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Cell Parameter Differences')
        
        # 5. Bond length differences
        if 'bond_diff' in valid_df.columns:
            bond_data = valid_df['bond_diff'].dropna()
            if len(bond_data) > 0:
                axes[1, 1].hist(bond_data, bins=30, alpha=0.7, color='plum', edgecolor='black')
                axes[1, 1].set_xlabel('Average Bond Length Difference (Å)')
                axes[1, 1].set_ylabel('Number of Structures')
                axes[1, 1].set_title('Bond Length Differences')
        
        # 6. Relaxation comparison (if original structures available)
        if 'rmsd_original_m3gnet' in valid_df.columns and 'rmsd_original_deeprelax' in valid_df.columns:
            relax_data = valid_df[['rmsd_original_m3gnet', 'rmsd_original_deeprelax']].dropna()
            if len(relax_data) > 0:
                axes[1, 2].scatter(relax_data['rmsd_original_m3gnet'], relax_data['rmsd_original_deeprelax'], alpha=0.6)
                min_rmsd = min(relax_data['rmsd_original_m3gnet'].min(), relax_data['rmsd_original_deeprelax'].min())
                max_rmsd = max(relax_data['rmsd_original_m3gnet'].max(), relax_data['rmsd_original_deeprelax'].max())
                axes[1, 2].plot([min_rmsd, max_rmsd], [min_rmsd, max_rmsd], 'r--', label='Equal Relaxation')
                axes[1, 2].set_xlabel('M3GNet Relaxation Distance (Å)')
                axes[1, 2].set_ylabel('DeepRelax Relaxation Distance (Å)')
                axes[1, 2].set_title('Relaxation Distance Comparison')
                axes[1, 2].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plots saved to {save_path}")
        
        plt.show()
    
    def generate_summary_report(self, df, save_path=None):
        """
        Generate a detailed summary report.
        """
        # Check if we have the required column
        if 'rmsd_m3gnet_deeprelax' not in df.columns:
            report = f"""
# M3GNet vs DeepRelax Relaxation Comparison Report

## Error: No RMSD data available
- No successful comparisons were completed
- Available columns: {list(df.columns)}
- Total entries: {len(df)}

## Debugging Information
Please check:
1. File naming conventions match between directories
2. Files can be read by ASE
3. Structures have compatible formats
"""
            print(report)
            if save_path:
                with open(save_path, 'w') as f:
                    f.write(report)
            return report
        
        valid_df = df.dropna(subset=['rmsd_m3gnet_deeprelax'])
        
        if len(valid_df) == 0:
            report = f"""
# M3GNet vs DeepRelax Relaxation Comparison Report

## Error: No valid RMSD calculations
- {len(df)} comparison attempts made
- All comparisons failed or returned NaN values
- Check error messages in console output
"""
            print(report)
            if save_path:
                with open(save_path, 'w') as f:
                    f.write(report)
            return report
        
        report = f"""
# M3GNet vs DeepRelax Relaxation Comparison Report

## Dataset Summary
- Total structures compared: {len(valid_df)}
- Average number of atoms per structure: {valid_df['n_atoms'].mean():.1f}
- Structure size range: {valid_df['n_atoms'].min()} - {valid_df['n_atoms'].max()} atoms

## Position Differences (RMSD)
- Mean RMSD: {valid_df['rmsd_m3gnet_deeprelax'].mean():.4f} ± {valid_df['rmsd_m3gnet_deeprelax'].std():.4f} Å
- Median RMSD: {valid_df['rmsd_m3gnet_deeprelax'].median():.4f} Å
- 95th percentile: {valid_df['rmsd_m3gnet_deeprelax'].quantile(0.95):.4f} Å
- Structures with RMSD > 0.5 Å: {(valid_df['rmsd_m3gnet_deeprelax'] > 0.5).sum()} ({100*(valid_df['rmsd_m3gnet_deeprelax'] > 0.5).mean():.1f}%)

## Volume Differences
"""
        
        if 'volume_percent_diff' in valid_df.columns:
            report += f"""- Mean volume difference: {valid_df['volume_percent_diff'].mean():.2f} ± {valid_df['volume_percent_diff'].std():.2f}%
- Median volume difference: {valid_df['volume_percent_diff'].median():.2f}%
- Structures with >5% volume difference: {(valid_df['volume_percent_diff'] > 5).sum()} ({100*(valid_df['volume_percent_diff'] > 5).mean():.1f}%)
"""
        
        if 'bond_diff' in valid_df.columns:
            bond_data = valid_df['bond_diff'].dropna()
            if len(bond_data) > 0:
                report += f"""
## Bond Length Differences
- Mean bond length difference: {bond_data.mean():.4f} ± {bond_data.std():.4f} Å
- Median bond length difference: {bond_data.median():.4f} Å
"""
        
        if 'rmsd_original_m3gnet' in valid_df.columns:
            m3g_relax = valid_df['rmsd_original_m3gnet'].dropna()
            deep_relax = valid_df['rmsd_original_deeprelax'].dropna()
            if len(m3g_relax) > 0 and len(deep_relax) > 0:
                report += f"""
## Relaxation Distances from Original
- M3GNet average relaxation: {m3g_relax.mean():.4f} ± {m3g_relax.std():.4f} Å
- DeepRelax average relaxation: {deep_relax.mean():.4f} ± {deep_relax.std():.4f} Å
- Correlation between relaxation distances: {valid_df[['rmsd_original_m3gnet', 'rmsd_original_deeprelax']].corr().iloc[0,1]:.3f}
"""
        
        report += f"""
## Top 10 Most Different Structures (by RMSD)
"""
        top_different = valid_df.nlargest(10, 'rmsd_m3gnet_deeprelax')[['structure_id', 'rmsd_m3gnet_deeprelax', 'formula', 'n_atoms']]
        report += top_different.to_string(index=False)
        
        print(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"\nReport saved to {save_path}")
        
        return report

# Example usage function
def compare_relaxations(m3gnet_dir, deeprelax_dir, original_dir=None, output_dir="comparison_results"):
    """
    Complete workflow to compare M3GNet and DeepRelax relaxations.
    
    Parameters:
    -----------
    m3gnet_dir : str
        Directory with M3GNet relaxed structures
    deeprelax_dir : str  
        Directory with DeepRelax relaxed structures
    original_dir : str, optional
        Directory with original unrelaxed structures
    output_dir : str
        Directory to save comparison results
    """
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize comparator
    comparator = RelaxationComparator(m3gnet_dir, deeprelax_dir, original_dir)
    
    # Run comparison
    print("Starting relaxation comparison...")
    df = comparator.compare_all()
    
    # Save results
    results_csv = output_path / "relaxation_comparison.csv"
    df.to_csv(results_csv, index=False)
    print(f"Results saved to: {results_csv}")
    
    # Only generate plots and reports if we have valid data
    if 'rmsd_m3gnet_deeprelax' in df.columns and len(df.dropna(subset=['rmsd_m3gnet_deeprelax'])) > 0:
        # Generate plots
        plot_path = output_path / "comparison_plots.png"
        comparator.plot_comparison_summary(df, save_path=plot_path)
        
        # Generate report
        report_path = output_path / "comparison_report.txt"
        comparator.generate_summary_report(df, save_path=report_path)
    else:
        print("⚠️  No valid comparisons available - skipping plots and detailed report")
        # Still generate error report
        report_path = output_path / "error_report.txt"
        comparator.generate_summary_report(df, save_path=report_path)
    
    print(f"\nComparison complete! Results saved to: {output_path}")
    
    return df, comparator

if __name__ == "__main__":
    # Example usage
    m3gnet_folder = "/m3gnet"
    deeprelax_folder = "/deeprelax"
    original_folder = "/originals" # Optional
    
    df, comparator = compare_relaxations(
        m3gnet_dir=m3gnet_folder,
        deeprelax_dir=deeprelax_folder,
        original_dir=original_folder,
        output_dir="relaxation_comparison_results"
    )
    
    # Quick statistics
    if 'rmsd_m3gnet_deeprelax' in df.columns:
        valid_df = df.dropna(subset=['rmsd_m3gnet_deeprelax'])
        if len(valid_df) > 0:
            print(f"\nQuick Summary:")
            print(f"Mean RMSD: {valid_df['rmsd_m3gnet_deeprelax'].mean():.4f} Å")
            print(f"Median RMSD: {valid_df['rmsd_m3gnet_deeprelax'].median():.4f} Å")
            if 'volume_percent_diff' in valid_df.columns:
                print(f"Mean volume difference: {valid_df['volume_percent_diff'].mean():.2f}%")
        else:
            print("\n❌ No valid RMSD calculations available")
    else:
        print("\n❌ No successful comparisons - check file matching and formats")
