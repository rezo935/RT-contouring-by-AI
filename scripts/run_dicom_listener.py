"""CLI script for running the DICOM Storage SCP listener."""

import argparse
import logging
import signal
import socket
import sys
from pathlib import Path

from src.dicom_listener.storage_scp import DICOMReceiver


def get_local_ip() -> str:
    """Get the local IP address of the machine.
    
    Returns:
        Local IP address as a string, or 'localhost' if unable to determine
    """
    try:
        # Connect to an external host to determine local IP
        # No actual connection is made, just to get the local interface
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "localhost"


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def print_banner(receiver: DICOMReceiver, local_ip: str = None):
    """Print startup banner with configuration information.
    
    Args:
        receiver: The DICOMReceiver instance
        local_ip: Local IP address to display in configuration
    """
    if local_ip is None:
        local_ip = get_local_ip()
    
    print("=" * 60)
    print("  DICOM Storage SCP (Listener)")
    print("=" * 60)
    print(f"  AE Title:    {receiver.ae_title}")
    print(f"  Port:        {receiver.port}")
    print(f"  Output Dir:  {receiver.output_dir}")
    print("=" * 60)
    print("  Configure in Eclipse:")
    print(f"    AE Title:  {receiver.ae_title}")
    print(f"    IP:        {local_ip}")
    print(f"    Port:      {receiver.port}")
    print("=" * 60)
    print("  Waiting for connections... (Ctrl+C to stop)")
    print()


def print_statistics(receiver: DICOMReceiver):
    """Print receiver statistics.
    
    Args:
        receiver: The DICOMReceiver instance
    """
    stats = receiver.get_stats()
    
    print("\n" + "=" * 60)
    print("  DICOM Receiver Statistics")
    print("=" * 60)
    print(f"  Uptime:           {stats['uptime']}")
    print(f"  Files Received:   {stats['files_received']}")
    print(f"    CT Images:      {stats['ct_images']}")
    print(f"    RT Structures:  {stats['rt_structs']}")
    print(f"  Unique Patients:  {stats['unique_patients']}")
    print(f"  Errors:           {stats['errors']}")
    print("=" * 60)


def signal_handler(signum, frame, receiver):
    """Handle Ctrl+C gracefully.
    
    Args:
        signum: Signal number
        frame: Current stack frame
        receiver: The DICOMReceiver instance
    """
    print("\n\nShutting down DICOM listener...")
    print_statistics(receiver)
    print("\nGoodbye!")
    sys.exit(0)


def main():
    """Main entry point for DICOM listener CLI."""
    parser = argparse.ArgumentParser(
        description="DICOM Storage SCP for receiving CT and RTSTRUCT files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--ae-title",
        type=str,
        default="AUTOCONTOUR",
        help="AE Title for this DICOM SCP"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=11112,
        help="Port number to listen on"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for received files (default: platform-dependent, typically ~/RadiotherapyData/DICOM_exports)"
    )
    
    parser.add_argument(
        "--ip",
        type=str,
        default=None,
        help="Local IP address to display in configuration banner (auto-detected if not specified)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize receiver
        receiver = DICOMReceiver(
            ae_title=args.ae_title,
            port=args.port,
            output_dir=args.output
        )
        
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, lambda s, f: signal_handler(s, f, receiver))
        
        # Print banner
        print_banner(receiver, args.ip)
        
        # Start the receiver (blocking)
        receiver.start(blocking=True)
        
    except Exception as e:
        logger.error(f"Failed to start DICOM listener: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
