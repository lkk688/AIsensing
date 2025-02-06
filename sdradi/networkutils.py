import subprocess
import sys
import socket
import ipaddress
import platform
import re

def check_ip_accessibility(ip_address):
    try:
        # Use the ping command to check if the IP is accessible
        # On Windows, use 'ping -n 1 <ip_address>'
        # On Linux/Mac, use 'ping -c 1 <ip_address>'
        response = subprocess.run(['ping', '-c', '1', ip_address], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Check the return code to determine if the ping was successful
        if response.returncode == 0:
            print(f"IP address {ip_address} is accessible.")
        else:
            print(f"IP address {ip_address} is not accessible.")
            sys.exit(1)  # Stop the program with an error code
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)  # Stop the program with an error code
        
def get_local_ip_ping(returnall=True):
    """Get the local IP address of the machine."""
    try:
        # Create a socket to get the local IP address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # Connect to a public DNS server
        local_ips = s.getsockname()
        s.close()
        
        # Filter out loopback and non-IPv4 addresses
        local_ips = [str(ip) for ip in local_ips] #convert all numbers to strings
        valid_ips = [ip for ip in local_ips if not ip.startswith("127.") and "." in ip]
        if returnall==True:
            return valid_ips
        else:
            return valid_ips[0]
    except Exception as e:
        print(f"Error getting local IP address: {e}")
        sys.exit(1)

def get_local_ip_socket(returnall=True):
    """Get the local IP address of the machine using network interfaces."""
    try:
        # Get the hostname of the machine
        hostname = socket.gethostname()
        
        # Get all IP addresses associated with the hostname
        ip_addresses = socket.gethostbyname_ex(hostname)[2]
        
        # Filter out loopback and non-IPv4 addresses
        valid_ips = [ip for ip in ip_addresses if not ip.startswith("127.") and "." in ip]
        
        if not valid_ips:
            raise Exception("No valid IPv4 address found.")
        
        if returnall==True:
            return valid_ips
        else:
            return valid_ips[0]
            # Return the first valid IP address
    except Exception as e:
        print(f"Error getting local IP address: {e}")
        sys.exit(1)

def get_local_ip(returnall=True):
    """Get the local IP address using ifconfig (Linux/Mac) or ipconfig (Windows)."""
    system = platform.system().lower()
    try:
        if system == "windows":
            # Run ipconfig and parse the output
            result = subprocess.run(['ipconfig'], stdout=subprocess.PIPE, text=True)
            output = result.stdout
            # Use regex to find all IPv4 addresses
            matches = re.finditer(r"IPv4 Address[ .]*: (\d+\.\d+\.\d+\.\d+)", output)
        else:
            # Run ifconfig and parse the output
            result = subprocess.run(['ifconfig'], stdout=subprocess.PIPE, text=True)
            output = result.stdout
            # Use regex to find all IPv4 addresses
            matches = re.finditer(r"inet (\d+\.\d+\.\d+\.\d+)", output)

        # Filter out loopback addresses (127.x.x.x)
        valid_ips = []
        for match in matches:
            ip = match.group(1)
            if not ip.startswith("127.") and "." in ip:
                valid_ips.append(ip)

        if not valid_ips:
            raise Exception("No valid IPv4 address found.")
        
        if returnall==True:
            return valid_ips
        else:
            return valid_ips[0]
            # Return the first valid IP address
    except Exception as e:
        print(f"Error getting local IP address: {e}")
        sys.exit(1)
        
def get_ping_command(ip):
    """Return the appropriate ping command based on the operating system."""
    system = platform.system().lower()
    if system == "windows":
        return ['ping', '-n', '1', '-w', '1000', ip]  # Windows uses '-n' for count and '-w' for timeout
    elif system in ["linux", "darwin"]:  # 'darwin' is macOS
        return ['ping', '-c', '1', '-W', '1', ip]  # Linux/Mac uses '-c' for count and '-W' for timeout
    else:
        print(f"Unsupported operating system: {system}")
        sys.exit(1)

def get_hostname(ip):
    """Get the hostname for a reachable IP address."""
    try:
        # Use socket.gethostbyaddr to resolve the IP to a hostname
        hostname, _, _ = socket.gethostbyaddr(ip)
        return hostname
    except socket.herror as e:
        print(f"Could not resolve hostname for IP {ip}: {e}")
        return None

def is_ip_reachable(ip):
    """Check if the IP address is reachable using ping."""
    try:
        # Get the appropriate ping command for the OS
        ping_command = get_ping_command(ip)

        # Use the ping command to check if the IP is reachable
        response = subprocess.run(
            ping_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        return response.returncode == 0
    except Exception as e:
        print(f"Error checking IP {ip}: {e}")
        return False
    
def scan_nearby_ips(base_ip, start_range=1, end_range=255):
    """Scan nearby IPs in the same subnet."""
    reachable_ips = []

    # Convert the base IP to an IPv4Address object
    try:
        network = ipaddress.ip_network(f"{base_ip}/24", strict=False)
    except ValueError as e:
        print(f"Invalid IP address or subnet: {e}")
        sys.exit(1)

    print(f"Scanning nearby IPs in the subnet: {network}")

    for ip in network.hosts():
        if str(ip) == base_ip:
            continue  # Skip the base IP itself
        ip=str(ip)
        if is_ip_reachable(ip):
            print(f"IP {ip} is reachable.")
            reachable_ips.append(ip)
            hostname = get_hostname(ip)
            if hostname:
                print(f"Hostname for IP {ip}: {hostname}")
            else:
                print(f"No hostname found for IP {ip}.")

    if not reachable_ips:
        print("No nearby IPs are reachable.")

    return reachable_ips

if __name__ == "__main__":
    # Get the local IP address or use a provided hostname/IP
    local_ip = get_local_ip(returnall=False)
    local_ip2 = get_local_ip_ping(returnall=False)
    local_ip3 = get_local_ip_socket(returnall=False)
    print(f"Local IP address: {local_ip}")

    # Scan nearby IPs in the same subnet
    reachable_ips = scan_nearby_ips(local_ip)

    print("Reachable IPs:")
    for ip in reachable_ips:
        print(ip)