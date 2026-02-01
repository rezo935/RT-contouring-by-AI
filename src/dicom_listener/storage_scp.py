"""DICOM Storage SCP implementation using pynetdicom."""

import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

import pydicom
from pynetdicom import AE, evt, AllStoragePresentationContexts, VerificationPresentationContexts
from pynetdicom.sop_class import (
    CTImageStorage,
    RTStructureSetStorage,
    Verification
)


logger = logging.getLogger(__name__)


class DICOMReceiver:
    """DICOM Storage SCP for receiving CT and RTSTRUCT files.
    
    This class implements a DICOM Storage Service Class Provider (SCP) that can
    receive CT Image Storage and RT Structure Set Storage SOP classes from
    DICOM sources like Eclipse TPS.
    
    Features:
    - Supports CT Image Storage and RT Structure Set Storage
    - C-ECHO verification support
    - Auto-organizes files by PatientID and StudyDate
    - Configurable file naming (CT_XXXX.dcm for CT, RS_label.dcm for RTSTRUCT)
    - Statistics tracking
    
    Args:
        ae_title: AE Title for this SCP (default: "AUTOCONTOUR")
        port: Port to listen on (default: 11112)
        output_dir: Directory to store received files (default: platform-dependent;
                   C:\\Users\\IBA\\RadiotherapyData\\DICOM_exports on Windows IBA workstations,
                   ~/RadiotherapyData/DICOM_exports elsewhere)
    """
    
    def __init__(
        self,
        ae_title: str = "AUTOCONTOUR",
        port: int = 11112,
        output_dir: Optional[str] = None
    ):
        """Initialize the DICOM Storage SCP.
        
        Args:
            ae_title: Application Entity title
            port: Port number to listen on
            output_dir: Output directory path for received files
        """
        self.ae_title = ae_title
        self.port = port
        
        # Set default output directory
        if output_dir is None:
            # Use platform-independent default path
            if Path(r"C:\Users\IBA\RadiotherapyData").exists():
                # Windows default for IBA workstation
                output_dir = r"C:\Users\IBA\RadiotherapyData\DICOM_exports"
            else:
                # Generic default for other systems
                output_dir = Path.home() / "RadiotherapyData" / "DICOM_exports"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            "files_received": 0,
            "ct_images": 0,
            "rt_structs": 0,
            "unique_patients": set(),
            "start_time": None,
            "errors": 0
        }
        
        # Initialize Application Entity
        self.ae = AE(ae_title=self.ae_title)
        
        # Define common transfer syntaxes for maximum compatibility
        transfer_syntaxes = [
            '1.2.840.10008.1.2',      # Implicit VR Little Endian
            '1.2.840.10008.1.2.1',    # Explicit VR Little Endian
            '1.2.840.10008.1.2.2',    # Explicit VR Big Endian
            '1.2.840.10008.1.2.4.50', # JPEG Baseline
            '1.2.840.10008.1.2.4.51', # JPEG Extended
            '1.2.840.10008.1.2.4.70', # JPEG Lossless
            '1.2.840.10008.1.2.4.90', # JPEG 2000 Lossless
            '1.2.840.10008.1.2.4.91', # JPEG 2000
            '1.2.840.10008.1.2.5',    # RLE Lossless
        ]
        
        # Add supported presentation contexts
        # Support CT Image Storage with all transfer syntaxes
        self.ae.add_supported_context(CTImageStorage, transfer_syntaxes)
        
        # Support RT Structure Set Storage with all transfer syntaxes
        self.ae.add_supported_context(RTStructureSetStorage, transfer_syntaxes)
        
        # Support Verification (C-ECHO)
        self.ae.add_supported_context(Verification)
        
        logger.info(f"DICOM Receiver initialized")
        logger.info(f"  AE Title: {self.ae_title}")
        logger.info(f"  Port: {self.port}")
        logger.info(f"  Output Directory: {self.output_dir}")
    
    def start(self, blocking: bool = True):
        """Start the DICOM Storage SCP.
        
        Args:
            blocking: If True, blocks until server is stopped. If False, runs in background.
        """
        self.stats["start_time"] = datetime.now()
        
        # Set up event handlers
        handlers = [
            (evt.EVT_C_STORE, self._handle_store),
            (evt.EVT_C_ECHO, self._handle_echo)
        ]
        
        logger.info(f"Starting DICOM Storage SCP on port {self.port}...")
        logger.info(f"Waiting for connections... (Ctrl+C to stop)")
        
        # Start listening
        self.ae.start_server(
            ('', self.port),
            block=blocking,
            evt_handlers=handlers
        )
    
    def _handle_store(self, event) -> int:
        """Handle incoming C-STORE request.
        
        Args:
            event: The C-STORE event
            
        Returns:
            Status code (0x0000 for success)
        """
        try:
            # Get the dataset
            ds = event.dataset
            ds.file_meta = event.file_meta
            
            # Extract patient and study information
            patient_id = str(ds.get("PatientID", "Unknown"))
            study_date = str(ds.get("StudyDate", "Unknown"))
            sop_class_uid = ds.SOPClassUID
            
            # Clean patient ID and study date for filesystem
            patient_id_clean = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in patient_id)
            study_date_clean = "".join(c if c.isdigit() else '' for c in study_date)
            
            # Create patient/study directory
            patient_dir = self.output_dir / patient_id_clean / study_date_clean
            patient_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine file type and generate filename
            if sop_class_uid == CTImageStorage:
                # CT Image - use instance number for ordering
                instance_num = ds.get("InstanceNumber", 0)
                filename = f"CT_{instance_num:04d}.dcm"
                self.stats["ct_images"] += 1
                file_type = "CT"
            elif sop_class_uid == RTStructureSetStorage:
                # RT Structure Set - use structure set label or default name
                struct_label = ds.get("StructureSetLabel", "label")
                struct_label_clean = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in struct_label)
                filename = f"RS_{struct_label_clean}.dcm"
                self.stats["rt_structs"] += 1
                file_type = "RTSTRUCT"
            else:
                # Other SOP class
                filename = f"DICOM_{ds.SOPInstanceUID}.dcm"
                file_type = "Other"
            
            # Save the file
            output_path = patient_dir / filename
            ds.save_as(output_path, write_like_original=False)
            
            # Update statistics
            self.stats["files_received"] += 1
            self.stats["unique_patients"].add(patient_id_clean)
            
            logger.info(f"Received {file_type}: {patient_id} / {study_date} -> {output_path}")
            
            # Return success status
            return 0x0000
            
        except Exception as e:
            logger.error(f"Error handling C-STORE: {e}", exc_info=True)
            self.stats["errors"] += 1
            # Return failure status
            return 0xC000
    
    def _handle_echo(self, event) -> int:
        """Handle incoming C-ECHO request.
        
        Args:
            event: The C-ECHO event
            
        Returns:
            Status code (0x0000 for success)
        """
        logger.debug("Received C-ECHO request")
        return 0x0000
    
    def get_stats(self) -> Dict:
        """Get receiver statistics.
        
        Returns:
            Dictionary with statistics:
                - files_received: Total number of files received
                - ct_images: Number of CT images received
                - rt_structs: Number of RT structures received
                - unique_patients: Number of unique patients
                - errors: Number of errors
                - uptime: Time since server started
        """
        stats = self.stats.copy()
        stats["unique_patients"] = len(self.stats["unique_patients"])
        
        if stats["start_time"]:
            uptime = datetime.now() - stats["start_time"]
            stats["uptime"] = str(uptime).split('.')[0]  # Remove microseconds
        else:
            stats["uptime"] = "Not started"
        
        return stats
