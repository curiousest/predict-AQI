If you ever have to do a docker volume on external HD, have an nfts partition mounted like this. exfat doesn't allow chmod/chown commands to be run, so you can't use some containers if not nfts:
http://askubuntu.com/questions/11840/how-do-i-use-chmod-on-an-ntfs-or-fat32-partition

```
For NTFS partitions, use the permissions option in fstab.

First unmount the ntfs partition.

Identify your partition UUID with blkid

sudo blkid
Then edit /etc/fstab

# Graphical 
gksu gedit /etc/fstab

# Command line
sudo -e /etc/fstab
And add or edit a line for the ntfs partition

    # change the "UUID" to your partition UUID
    UUID=12102C02102CEB83 /media/windows ntfs-3g auto,users,permissions 0 0
Make a mount point (if needed)

sudo mkdir /media/windows
Now mount the partition

mount /media/windows
The options I gave you, auto, will automatically mount the partition when you boot and users allows users to mount and umount .

You can then use chown and chmod on the ntfs partition.
```

