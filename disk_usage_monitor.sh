#!/bin/bash

# Check disk usage and alert if usage exceeds 80%
threshold=80

# Get disk usage
df -H | awk 'NR>1 { if ($5+0 > threshold) print $0; }' threshold="$threshold" | while read line; do
    echo "Alert: Disk usage exceeded 80%: $line"
done