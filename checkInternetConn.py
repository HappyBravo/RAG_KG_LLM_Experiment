# Test if an internet connection is present
import socket
REMOTE_SERVER = "one.one.one.one"
TIME_OUT = 30

def is_connected(hostname = REMOTE_SERVER, time_out = TIME_OUT):
  try:
    # See if we can resolve the host name - tells us if there is
    # A DNS listening
    host = socket.gethostbyname(hostname)
    # Connect to the host - tells us if the host is actually reachable
    s = socket.create_connection((host, 80), 30)
    s.close()
    return True
  except Exception as e:
     print("Error :", e)
     # pass # We ignore any errors, returning False
  return False

if __name__ == "__main__":
    print(is_connected(REMOTE_SERVER))