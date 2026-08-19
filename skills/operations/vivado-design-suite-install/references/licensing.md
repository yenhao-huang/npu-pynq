# Vivado Basic Tier certificate licensing

If the AMD licensing portal presents the listed certificate products, use:

```text
Vivado Basic Tier License, Node Locked License
```

Do not select the 60-day Vivado Enterprise evaluation for a PYNQ-Z1-only workflow.

## Host registration

The license host is the Windows computer on which Vivado will run.

- Host name: the Windows PC name (a label for recognition).
- Operating system: Windows 64-bit for a current 64-bit Windows installation.
- Host ID type: Ethernet MAC.
- Host ID value: the 12-character MAC/Host ID for a physical network adapter, without separators when the portal requires that format.

Prefer the Host ID shown by Vivado License Manager's **System Information** view. Avoid VPN, virtual-machine, WSL/Hyper-V, Bluetooth, and Wi-Fi Direct adapters. A license certificate is bound to its Host ID, so a changed network adapter or disk can require rehosting.

After the portal generates the `.lic` file, import it in Vivado with **Help → Manage License**, then use **Load License** or **Copy License**. Verify the license status before starting a design.

Sources: [AMD licensing FAQ](https://www.amd.com/en/products/software/adaptive-socs-and-fpgas/licensing-faq.html), [AMD UG973: Create and Generate a License Key File](https://docs.amd.com/r/2025.2-English/ug973-vivado-release-notes-install-license/Create-and-Generate-a-License-Key-File).
