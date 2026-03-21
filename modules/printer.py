import subprocess
import logging
import os
import glob

logger = logging.getLogger(__name__)

class ZebraPrinter:
    def __init__(self, device_path='/dev/usb/lp0'):
        self.device_path = device_path
        self.use_pyusb = False
        if not os.path.exists(self.device_path):
            usb_printers = glob.glob('/dev/usb/lp*')
            if usb_printers:
                self.device_path = usb_printers[0]
                logger.info(f"Default printer {device_path} not found. Auto-detected {self.device_path}")
            else:
                logger.warning(f"No USB printers found in /dev/usb/lp*. Enabling PyUSB fallback.")
                self.use_pyusb = True

    def print_pallet_qr(self, pallet_id):
        if not pallet_id:
            logger.warning("No pallet ID provided for printing.")
            return False, "No pallet ID provided"
            
        logger.info(f"Generating ZPL for Pallet ID: {pallet_id}")
        
        # Using the exact ZPL from your successful terminal test
        # ^BQN,2,15 = QR code scale 15. ^A0N,40,40 = Text font.
        zpl_command = f"^XA^FO50,50^BQN,2,15^FDQA,{pallet_id}^FS^FO50,320^A0N,40,40^FDPallet: {pallet_id}^FS^XZ"
        
        if self.use_pyusb:
            return self._print_via_pyusb(zpl_command)
        
        try:
            # Method 1: Try direct write (Works if user is added to 'lp' group)
            with open(self.device_path, 'wb') as printer:
                printer.write(zpl_command.encode('utf-8'))
            logger.info("Successfully printed directly to USB.")
            return True, ""
            
        except PermissionError:
            # Method 2: Fallback to the sudo tee method you verified in terminal
            logger.warning(f"Permission denied for {self.device_path}. Falling back to sudo tee.")
            try:
                cmd = f'echo "{zpl_command}" | sudo tee {self.device_path} > /dev/null'
                subprocess.run(cmd, shell=True, check=True)
                logger.info("Successfully printed using sudo tee fallback.")
                return True, ""
            except Exception as e:
                logger.error(f"Fallback printing failed: {e}")
                return False, f"Fallback printing failed: {e}"
                
        except Exception as e:
            logger.error(f"Printer error: {e}")
            return False, str(e)

    def _print_via_pyusb(self, zpl_command):
        try:
            import usb.core
            import usb.util
            
            # Find Zebra printer by vendor ID (0x0a5f)
            dev = usb.core.find(idVendor=0x0a5f)
            if dev is None:
                logger.error("No Zebra printer found via PyUSB.")
                return False, "No printer found via PyUSB"
                
            # Detach kernel driver if active
            if dev.is_kernel_driver_active(0):
                try:
                    dev.detach_kernel_driver(0)
                except usb.core.USBError as e:
                    logger.warning(f"Could not detach kernel driver: {e}")
                    
            try:
                dev.set_configuration()
            except usb.core.USBError as e:
                logger.warning(f"Could not set configuration: {e}")
                
            cfg = dev.get_active_configuration()
            intf = cfg[(0,0)]
            
            ep = usb.util.find_descriptor(
                intf,
                custom_match = lambda e: usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_OUT
            )
            
            if ep is None:
                logger.error("Endpoint not found via PyUSB.")
                return False, "Endpoint not found"
                
            ep.write(zpl_command.encode('utf-8'))
            logger.info("Successfully printed using PyUSB fallback.")
            return True, ""
            
        except ImportError:
            logger.error("PyUSB not installed. Cannot use USB fallback. Please run: pip install pyusb")
            return False, "PyUSB not installed"
        except Exception as e:
            logger.error(f"PyUSB printing failed: {e}")
            if "Access denied" in str(e) or "Insufficient permissions" in str(e):
                logger.error("Try running the app with 'sudo' or setting udev rules for the printer.")
            return False, f"PyUSB failed: {e}"