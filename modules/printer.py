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

        # Clean ZPL command (no leading/trailing whitespace)
        # ^BQN,2,25: N=Normal, 2=Model 2, 25=Magnification (Extra Large QR)
        # ^FO50,1000: Positioned lower to ensure no overlap even with extra large 25x QR codes
        zpl_command = (
            "^XA\n"
            f"^FO50,50^BQN,2,25^FDQA,{pallet_id}^FS\n"
            f"^FO50,1000^A0N,60,60^FDPallet: {pallet_id}^FS\n"
            "^XZ"
        )

        if self.use_pyusb:
            return self._print_via_pyusb(zpl_command)

        try:
            # Method 1: Direct write (works if user is in 'lp' group)
            with open(self.device_path, 'wb') as printer:
                printer.write(zpl_command.encode('utf-8'))
            logger.info("Successfully printed directly to USB.")
            return True, ""

        except PermissionError:
            # Method 2: Fallback to sudo tee
            logger.warning(f"Permission denied for {self.device_path}. Falling back to sudo tee.")
            try:
                subprocess.run(
                    ['sudo', 'tee', self.device_path],
                    input=zpl_command.encode('utf-8'),
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
                logger.info("Successfully printed using sudo tee fallback.")
                return True, ""
            except Exception as e:
                logger.exception("Fallback printing failed")
                return False, f"Fallback printing failed: {e}"

        except Exception as e:
            logger.exception("Printer error")
            return False, str(e)

    def _print_via_pyusb(self, zpl_command):
        try:
            import usb.core
            import usb.util
        except ImportError:
            logger.error("PyUSB not installed. Cannot use USB fallback. Please run: pip install pyusb")
            return False, "PyUSB not installed"

        dev = usb.core.find(idVendor=0x0a5f)
        if dev is None:
            logger.error("No Zebra printer found via PyUSB.")
            return False, "No printer found via PyUSB"

        return self._write_to_usb_device(dev, zpl_command)

    def _safe_reset(self, dev):
        try:
            dev.reset()
        except Exception as e:
            logger.warning(f"Device reset warning (non-fatal): {e}")

    def _detach_kernel_drivers(self, dev):
        detached_ifaces = []
        for iface_num in range(dev.get_active_configuration().bNumInterfaces):
            try:
                if dev.is_kernel_driver_active(iface_num):
                    dev.detach_kernel_driver(iface_num)
                    detached_ifaces.append(iface_num)
                    logger.debug(f"Detached kernel driver from interface {iface_num}")
            except Exception as e:
                logger.warning(f"Could not detach kernel driver from interface {iface_num}: {e}")
        return detached_ifaces

    def _safe_set_configuration(self, dev):
        try:
            dev.set_configuration()
        except Exception as e:
            logger.warning(f"set_configuration warning (non-fatal): {e}")

    def _find_out_endpoint(self, intf):
        import usb.util
        return usb.util.find_descriptor(
            intf,
            custom_match=lambda e: (
                usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_OUT
            ),
        )

    def _release_usb_resources(self, dev, interface_claimed, detached_ifaces):
        import usb.util
        if interface_claimed:
            try:
                cfg = dev.get_active_configuration()
                intf = cfg[(0, 0)]
                usb.util.release_interface(dev, intf.bInterfaceNumber)
                logger.debug("Interface released.")
            except Exception:
                pass

        for iface_num in detached_ifaces:
            try:
                dev.attach_kernel_driver(iface_num)
                logger.debug(f"Re-attached kernel driver to interface {iface_num}")
            except Exception as e:
                logger.warning(f"Could not re-attach kernel driver to interface {iface_num}: {e}")

        usb.util.dispose_resources(dev)
        logger.debug("USB resources disposed — printer closed.")

    def _write_to_usb_device(self, dev, zpl_command, _retry=True):
        """Write ZPL to USB device, retrying once with a full reset on [Errno 5]."""
        import usb.util

        interface_claimed = False
        detached_ifaces = []

        try:
            # 1. Reset device to clear any stale USB state
            self._safe_reset(dev)

            # 2. Detach kernel driver (usblp)
            detached_ifaces = self._detach_kernel_drivers(dev)
            self._safe_set_configuration(dev)

            cfg = dev.get_active_configuration()
            intf = cfg[(0, 0)]

            # 3. Claim the interface
            usb.util.claim_interface(dev, intf.bInterfaceNumber)
            interface_claimed = True

            ep = self._find_out_endpoint(intf)
            if ep is None:
                logger.error("Endpoint not found via PyUSB.")
                return False, "Endpoint not found"

            # 4. Send ZPL
            ep.write(zpl_command.encode('utf-8'), timeout=5000)
            logger.info("Successfully printed using PyUSB.")
            return True, ""

        except Exception as e:
            err_str = str(e)
            logger.exception("PyUSB printing failed")

            if "[Errno 5]" in err_str and _retry:
                # Hard reset and retry once on I/O error
                logger.warning("Got [Errno 5] I/O error — resetting device and retrying once...")
                self._safe_reset(dev)
                return self._write_to_usb_device(dev, zpl_command, _retry=False)

            if "Access denied" in err_str or "Insufficient permissions" in err_str:
                logger.error("Try running the app with 'sudo' or add a udev rule for the printer.")

            return False, f"PyUSB failed: {e}"

        finally:
            # 5. Release resources
            self._release_usb_resources(dev, interface_claimed, detached_ifaces)