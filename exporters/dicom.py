"""
DICOM Exporter

Exports simulated CT volumes as DICOM series with proper metadata
for compatibility with medical imaging software.
"""

from pathlib import Path
from datetime import datetime
from typing import Optional, Callable
import numpy as np

from core.windowing import linear_to_uint, map_window_to_uint_range

try:
    import pydicom
    from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
    from pydicom.uid import (
        generate_uid,
        ExplicitVRLittleEndian,
        CTImageStorage,
    )
    from pydicom.sequence import Sequence
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


class DICOMExporter:
    """
    Exports CT volumes as DICOM series.
    
    Creates a series of DICOM files, one per slice.

    Pixel encoding modes:
    - `mu_scaled_u16` (default): map mu(cm^-1) to unitless uint16 gray [0, 65535]
    - `hu_u16`: convert mu to HU and store HU+1024 in uint16
    """

    # Reference water attenuation used for HU conversion:
    # HU = 1000 * (mu / mu_water_ref - 1)
    MU_WATER_REF_CM_INV = 0.171

    @staticmethod
    def is_available() -> bool:
        """Whether pydicom runtime dependency is available."""
        return HAS_PYDICOM
    
    def __init__(
        self,
        patient_name: str = "Anonymous^Patient",
        patient_id: str = "SIMULATION001",
        study_description: str = "CT Simulation",
        series_description: str = "Simulated CT Series",
        institution_name: str = "Research Institution",
        manufacturer: str = "CT Simulation Software",
        pixel_value_mode: str = "mu_scaled_u16",
    ):
        """
        Initialize DICOM exporter with metadata.
        
        Args:
            patient_name: Patient name in DICOM format (Family^Given)
            patient_id: Patient ID
            study_description: Description of the study
            series_description: Description of the series
            institution_name: Name of institution
            manufacturer: Equipment manufacturer
            pixel_value_mode: `mu_scaled_u16` or `hu_u16`
        """
        if not HAS_PYDICOM:
            raise ImportError(
                "pydicom is required for DICOM export. "
                "Install it with: pip install pydicom"
            )
        
        self.patient_name = patient_name
        self.patient_id = patient_id
        self.study_description = study_description
        self.series_description = series_description
        self.institution_name = institution_name
        self.manufacturer = manufacturer
        if pixel_value_mode not in ("mu_scaled_u16", "hu_u16"):
            raise ValueError("pixel_value_mode must be 'mu_scaled_u16' or 'hu_u16'")
        self.pixel_value_mode = pixel_value_mode
        
        # Generate unique identifiers for the series
        self.study_instance_uid = generate_uid()
        self.series_instance_uid = generate_uid()
        self.frame_of_reference_uid = generate_uid()
        
        # Timestamp
        now = datetime.now()
        self.study_date = now.strftime("%Y%m%d")
        self.study_time = now.strftime("%H%M%S.%f")
        self.series_date = self.study_date
        self.series_time = self.study_time

    @staticmethod
    def _format_ds(value: float) -> str:
        """Format a float into DICOM DS-compatible text length."""
        return f"{float(value):.10g}"

    @classmethod
    def _mu_to_hu(cls, mu_value):
        """Convert linear attenuation (cm^-1) to Hounsfield Units."""
        return 1000.0 * (np.asarray(mu_value, dtype=np.float64) / cls.MU_WATER_REF_CM_INV - 1.0)
    
    def export(
        self,
        ct_volume: "CTVolume",
        output_dir: str | Path,
        window_center: float = 0.2,
        window_width: float = 0.4,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> list[Path]:
        """
        Export CT volume as DICOM series.
        
        Args:
            ct_volume: CTVolume object to export
            output_dir: Directory to save DICOM files
            window_center: Viewer window center in linear attenuation (cm^-1)
            window_width: Viewer window width in linear attenuation (cm^-1)
            progress_callback: Optional callback(progress: 0.0-1.0)
            
        Returns:
            List of paths to created DICOM files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get volume data
        volume = ct_volume.data
        num_slices = volume.shape[0]
        rows, cols = volume.shape[1], volume.shape[2]
        voxel_size = ct_volume.voxel_size
        origin = ct_volume.origin

        volume = np.asarray(volume, dtype=np.float64)

        if self.pixel_value_mode == "hu_u16":
            # Convert internal attenuation to HU (stored as HU + 1024).
            export_values = self._mu_to_hu(volume)
            rescale_slope = 1.0
            rescale_intercept = -1024.0
            rescale_type = "HU"
            window_center_export = float(self._mu_to_hu(window_center))
            window_width_export = max(1.0, float(1000.0 * window_width / self.MU_WATER_REF_CM_INV))
            if np.isnan(window_center_export):
                window_center_export = 0.5 * float(np.min(export_values) + np.max(export_values))
            if window_width <= 0:
                window_width_export = max(float(np.max(export_values) - np.min(export_values)), 1.0)
            window_explanation = "CT HU"
        else:
            # Default: scale mu directly into unitless uint16 grayscale [0, 65535].
            export_values = volume
            mu_min = float(np.min(export_values))
            mu_max = float(np.max(export_values))

            rescale_slope = 1.0
            rescale_intercept = 0.0
            rescale_type = "US"

            if mu_max > mu_min:
                window_center_export, window_width_export = map_window_to_uint_range(
                    center=float(window_center),
                    width=float(window_width),
                    source_min=mu_min,
                    source_max=mu_max,
                    output_dtype=np.uint16,
                    min_output_width=1.0,
                )
            else:
                window_center_export = 32768.0
                window_width_export = 65535.0
            window_explanation = "Scaled Gray 0-65535"
        
        created_files = []
        
        for i in range(num_slices):
            slice_values = export_values[i, :, :]

            # Convert export values to stored uint16.
            if self.pixel_value_mode == "hu_u16":
                stored_values = ((slice_values - rescale_intercept) / rescale_slope)
            else:
                if mu_max > mu_min:
                    stored_values = linear_to_uint(
                        slice_values,
                        lower=mu_min,
                        upper=mu_max,
                        output_dtype=np.uint16,
                    )
                else:
                    stored_values = np.zeros_like(slice_values, dtype=np.uint16)
            if self.pixel_value_mode == "hu_u16":
                stored_values = np.clip(stored_values, 0, 65535).astype(np.uint16)
            
            # Create DICOM dataset
            ds = self._create_dataset(
                slice_index=i,
                num_slices=num_slices,
                rows=rows,
                cols=cols,
                voxel_size=voxel_size,
                origin=origin,
                window_center=window_center_export,
                window_width=window_width_export,
                rescale_slope=rescale_slope,
                rescale_intercept=rescale_intercept,
                rescale_type=rescale_type,
                window_explanation=window_explanation,
            )
            
            # Set pixel data
            ds.PixelData = stored_values.tobytes()
            
            # Save file
            filename = output_dir / f"CT_{i:04d}.dcm"
            ds.save_as(filename, enforce_file_format=True)
            created_files.append(filename)
            
            # Report progress
            if progress_callback is not None:
                progress_callback((i + 1) / num_slices)
        
        return created_files
    
    def _create_dataset(
        self,
        slice_index: int,
        num_slices: int,
        rows: int,
        cols: int,
        voxel_size: float,
        origin: np.ndarray,
        window_center: float,
        window_width: float,
        rescale_slope: float,
        rescale_intercept: float,
        rescale_type: str,
        window_explanation: str,
    ) -> FileDataset:
        """Create a DICOM dataset for a single slice."""
        
        # Create file meta information
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = CTImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = generate_uid()
        file_meta.ImplementationVersionName = "CTSIM_1.0"
        
        # Create the FileDataset
        ds = FileDataset(
            filename_or_obj="",
            dataset={},
            file_meta=file_meta,
            preamble=b"\x00" * 128
        )
        
        # Patient Module
        ds.PatientName = self.patient_name
        ds.PatientID = self.patient_id
        ds.PatientBirthDate = ""
        ds.PatientSex = "O"  # Other
        
        # General Study Module
        ds.StudyInstanceUID = self.study_instance_uid
        ds.StudyDate = self.study_date
        ds.StudyTime = self.study_time
        ds.ReferringPhysicianName = ""
        ds.StudyID = "1"
        ds.AccessionNumber = ""
        ds.StudyDescription = self.study_description
        
        # General Series Module
        ds.SeriesInstanceUID = self.series_instance_uid
        ds.SeriesNumber = 1
        ds.Modality = "CT"
        ds.SeriesDate = self.series_date
        ds.SeriesTime = self.series_time
        ds.SeriesDescription = self.series_description
        ds.BodyPartExamined = ""
        ds.PatientPosition = "HFS"  # Head First Supine
        
        # Frame of Reference Module
        ds.FrameOfReferenceUID = self.frame_of_reference_uid
        ds.PositionReferenceIndicator = ""
        
        # General Equipment Module
        ds.Manufacturer = self.manufacturer
        ds.InstitutionName = self.institution_name
        ds.StationName = "SIMULATOR"
        ds.ManufacturerModelName = "CT Simulation v1.0"
        ds.SoftwareVersions = "1.0"
        
        # CT Image Module
        ds.KVP = "120"  # Simulated tube voltage
        ds.AcquisitionNumber = 1
        ds.ImageType = ["DERIVED", "PRIMARY", "AXIAL"]
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.Rows = rows
        ds.Columns = cols
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0  # Unsigned
        
        # Pixel spacing and slice information
        ds.PixelSpacing = [voxel_size, voxel_size]
        ds.SliceThickness = voxel_size
        ds.SpacingBetweenSlices = voxel_size
        
        # Image position (patient coordinates)
        slice_position_z = origin[2] + slice_index * voxel_size
        ds.ImagePositionPatient = [
            float(origin[0]),
            float(origin[1]),
            float(slice_position_z)
        ]
        
        # Image orientation (standard axial)
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        
        # Slice location
        ds.SliceLocation = float(slice_position_z)
        ds.InstanceNumber = slice_index + 1
        
        # SOP Instance
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        
        # Rescale metadata.
        ds.RescaleSlope = self._format_ds(rescale_slope)
        ds.RescaleIntercept = self._format_ds(rescale_intercept)
        ds.RescaleType = rescale_type
        
        # Window settings
        ds.WindowCenter = self._format_ds(window_center)
        ds.WindowWidth = self._format_ds(window_width)
        ds.WindowCenterWidthExplanation = window_explanation
        
        # Additional recommended tags
        ds.ContentDate = self.study_date
        ds.ContentTime = self.study_time
        ds.InstanceCreationDate = self.study_date
        ds.InstanceCreationTime = self.study_time
        
        # Image comments
        ds.ImageComments = (
            f"Simulated CT slice {slice_index + 1} of {num_slices}; "
            f"pixel_mode={self.pixel_value_mode}"
        )
        
        return ds
    
    def reset_uids(self) -> None:
        """Generate new UIDs for a new series."""
        self.study_instance_uid = generate_uid()
        self.series_instance_uid = generate_uid()
        self.frame_of_reference_uid = generate_uid()
        
        now = datetime.now()
        self.study_date = now.strftime("%Y%m%d")
        self.study_time = now.strftime("%H%M%S.%f")
        self.series_date = self.study_date
        self.series_time = self.study_time
