using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

namespace AirWriting
{
    public class AirWritingReceiver : MonoBehaviour
    {
        [Header("Network Settings")]
        [Tooltip("The UDP port to listen on. Matches python_to_unity in system.yaml")]
        public int listenPort = 12346;

        [Header("Debug")]
        public bool showDebugLogs = false;

        private UdpClient _udpClient;
        private Thread _receiveThread;
        private bool _isRunning;
        
        // Thread-safe data holding
        private readonly object _dataLock = new object();
        private AirWritingParsedData _latestData;

        // Events that other scripts can subscribe to
        public event Action<AirWritingParsedData> OnDataReceived;

        void Start()
        {
            StartReceiver();
        }

        private void StartReceiver()
        {
            try
            {
                _udpClient = new UdpClient(listenPort);
                _isRunning = true;
                _receiveThread = new Thread(ReceiveDataWorker)
                {
                    IsBackground = true,
                    Name = "AirWriting UDP Receiver"
                };
                _receiveThread.Start();
                
                Debug.Log($"[AirWriting] Listening for UDP on port {listenPort}");
            }
            catch (Exception e)
            {
                Debug.LogError($"[AirWriting] Failed to start UDP receiver: {e.Message}");
            }
        }

        private void ReceiveDataWorker()
        {
            IPEndPoint endPoint = new IPEndPoint(IPAddress.Any, listenPort);

            while (_isRunning)
            {
                try
                {
                    // Block until packet arrives
                    byte[] data = _udpClient.Receive(ref endPoint);
                    string json = Encoding.UTF8.GetString(data);

                    if (showDebugLogs)
                    {
                        Debug.Log($"[AirWriting RAW] {json}");
                    }

                    // Parse JSON on background thread
                    AirWritingPacket packet = JsonUtility.FromJson<AirWritingPacket>(json);
                    
                    if (packet != null && packet.t == "f")
                    {
                        var parsed = new AirWritingParsedData(packet);
                        
                        lock (_dataLock)
                        {
                            _latestData = parsed;
                        }
                    }
                }
                catch (SocketException e)
                {
                    // Ignore exact thread abort exceptions during shutdown
                    if (_isRunning)
                        Debug.LogWarning($"[AirWriting] Socket error: {e.Message}");
                }
                catch (Exception e)
                {
                    Debug.LogError($"[AirWriting] Parse error: {e.Message}");
                }
            }
        }

        void Update()
        {
            // Trigger events on Unity's main thread
            AirWritingParsedData dataToProcess = null;

            lock (_dataLock)
            {
                if (_latestData != null)
                {
                    dataToProcess = _latestData;
                    _latestData = null; // Clear so we don't process the same frame twice
                }
            }

            if (dataToProcess != null)
            {
                OnDataReceived?.Invoke(dataToProcess);
            }
        }

        void OnDisable()
        {
            _isRunning = false;
            
            if (_udpClient != null)
            {
                _udpClient.Close();
                _udpClient = null;
            }

            if (_receiveThread != null && _receiveThread.IsAlive)
            {
                _receiveThread.Join(500); // give it half a second to close
            }
        }
    }
}
