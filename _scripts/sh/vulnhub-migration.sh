VM=$1;
OS=$2;

if [ ! -z $VM ] && [ ! -z $OS ] && [ ! -z $OS ]; then
    LILVM=`echo $1 | tr 'A-Z' 'a-z'`; # Stapler -> stapler
    MAJOS=`echo "$(tr '[:lower:]' '[:upper:]' <<< ${OS:0:1})${OS:1}"`; # linux -> Linux
    # 1. Create img folder
    mkdir -p ~/workspace/amirr0r.github.io/assets/img/vulnhub/$OS/$LILVM
    # 2. Copy screenshots into img folder
    cp $VULNHUB/$VM/img/*.png ~/workspace/amirr0r.github.io/assets/img/vulnhub/$OS/$LILVM
    cp $VULNHUB/$VM/img/*.gif ~/workspace/amirr0r.github.io/assets/img/vulnhub/$OS/$LILVM
    # 3. Create WU post
    prefix=`date --rfc-3339=date`;
    wu_file="$HOME/workspace/amirr0r.github.io/_posts/$prefix-vulnhub-$LILVM.md";
    echo "---" > $wu_file;
    echo "title: VulnHub - $VM" >> $wu_file;
    echo "date: `date --rfc-3339=seconds | cut -d'+' -f1` +0100" >> $wu_file;
    echo "categories: [VulnHub walkthroughs, $MAJOS]" >> $wu_file;
    echo "tags: [vulnhub-$OS, writeup, oscp-prep]" >> $wu_file;
    echo "image: img/$VM.png" >> $wu_file;
    echo "---" >> $wu_file;
    tail --lines=+4 $VULNHUB/$VM/README.md >> $wu_file;
    sed -i "s/img/\/assets\/img\/vulnhub\/$OS\/$LILVM/g" $wu_file;
else
    echo "Usage: bash vulnhub-migration.sh <VM_name> <OS:linux|windows|...>"
fi