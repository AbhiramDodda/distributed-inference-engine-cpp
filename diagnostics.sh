#!/bin/bash

echo "=========================================="
echo "Distributed Inference System Diagnostics"
echo "=========================================="
echo ""

# Check if processes are running
echo "1. Process Status:"
echo "-------------------"
if pgrep -f worker_node > /dev/null; then
    echo "✓ Worker nodes running:"
    ps aux | grep worker_node | grep -v grep | awk '{print "  PID " $2 ": " $11 " " $12 " " $13}'
else
    echo "✗ No worker nodes running"
fi

if pgrep -f gateway > /dev/null; then
    echo "✓ Gateway running:"
    ps aux | grep gateway | grep -v grep | awk '{print "  PID " $2 ": " $11}'
else
    echo "✗ Gateway not running"
fi
echo ""

# Check ports
echo "2. Port Status:"
echo "---------------"
for port in 8000 8001 8002 8003; do
    if ss -tuln | grep ":$port " > /dev/null 2>&1; then
        echo "✓ Port $port is listening"
    else
        echo "✗ Port $port is NOT listening"
    fi
done
echo ""

# Test worker health endpoints
echo "3. Worker Health Checks:"
echo "------------------------"
for port in 8001 8002 8003; do
    echo "Testing localhost:$port/health..."
    response=$(curl -s -w "\n%{http_code}" http://localhost:$port/health 2>&1)
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n-1)
    
    if [ "$http_code" = "200" ]; then
        echo "✓ Worker on port $port is healthy"
        echo "  Response: $body" | head -c 100
        echo ""
    else
        echo "✗ Worker on port $port failed (HTTP $http_code)"
        echo "  Error: $body"
    fi
done
echo ""

# Test gateway stats
echo "4. Gateway Status:"
echo "------------------"
if curl -s http://localhost:8000/stats > /dev/null 2>&1; then
    stats=$(curl -s http://localhost:8000/stats)
    echo "✓ Gateway responding"
    echo "$stats" | python3 -m json.tool 2>/dev/null || echo "$stats"
else
    echo "✗ Gateway not responding"
fi
echo ""

# Test direct worker inference
echo "5. Direct Worker Test:"
echo "----------------------"
echo "Testing POST to localhost:8001/infer..."
test_payload='{"request_id": "diag_test", "input_data": [1.0, 2.0, 3.0, 4.0]}'
response=$(curl -s -w "\n%{http_code}" -X POST http://localhost:8001/infer \
    -H "Content-Type: application/json" \
    -d "$test_payload" 2>&1)
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | head -n-1)

if [ "$http_code" = "200" ]; then
    echo "✓ Direct worker inference successful"
    echo "  Response: $body" | head -c 200
    echo ""
else
    echo "✗ Direct worker inference failed (HTTP $http_code)"
    echo "  Error: $body"
fi
echo ""

# Test gateway inference
echo "6. Gateway Inference Test:"
echo "--------------------------"
echo "Testing POST to localhost:8000/infer..."
response=$(curl -s -w "\n%{http_code}" -X POST http://localhost:8000/infer \
    -H "Content-Type: application/json" \
    -d "$test_payload" 2>&1)
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | head -n-1)

if [ "$http_code" = "200" ]; then
    echo "✓ Gateway inference successful"
    echo "  Response: $body" | head -c 200
    echo ""
else
    echo "✗ Gateway inference failed (HTTP $http_code)"
    echo "  Error: $body"
fi
echo ""

echo "=========================================="
echo "Diagnostic Complete"
echo "=========================================="