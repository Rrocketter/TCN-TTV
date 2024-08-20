import matplotlib.pyplot as plt

# Data for epochs, training loss, and validation loss
epochs = list(range(1, 51))
train_loss = [0.1935, 0.1804, 0.1796, 0.1792, 0.1783, 0.1771, 0.1752, 0.1736, 0.1728, 0.1723,
              0.1717, 0.1707, 0.1703, 0.1700, 0.1697, 0.1692, 0.1689, 0.1688, 0.1688, 0.1685,
              0.1684, 0.1684, 0.1683, 0.1682, 0.1681, 0.1681, 0.1681, 0.1679, 0.1678, 0.1679,
              0.1679, 0.1677, 0.1679, 0.1678, 0.1678, 0.1677, 0.1677, 0.1678, 0.1677, 0.1676,
              0.1677, 0.1676, 0.1676, 0.1676, 0.1676, 0.1676]
val_loss = [0.1497, 0.1399, 0.1394, 0.1401, 0.1405, 0.1412, 0.1413, 0.1494, 0.1421, 0.1438,
            0.1441, 0.1443, 0.1465, 0.1479, 0.1478, 0.1467, 0.1478, 0.1567, 0.1664, 0.1525,
            0.1487, 0.1536, 0.4605, 0.1582, 0.1640, 0.1516, 0.1576, 0.1549, 0.1795, 0.1532,
            0.1821, 0.1711, 0.1536, 0.1912, 0.1504, 0.1609, 0.1654, 0.1538, 0.1646, 0.1901,
            0.1485, 0.1757, 0.1889, 0.1611, 0.1536, 0.1509, 0.1534, 0.1567, 0.1595, 0.1835]

# Plotting the graph
plt.figure(figsize=(12, 6))
plt.plot(epochs, train_loss, label='Training Loss', color='blue', marker='o')
plt.plot(epochs, val_loss, label='Validation Loss', color='red', marker='x')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss vs. Epoch')
plt.legend()
plt.grid(True)
plt.show()