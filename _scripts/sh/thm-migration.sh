VM=$1;
OS=$2;

if [ ! -z $VM ] && [ ! -z $OS ] && [ ! -z $OS ]; then
    LILVM=`echo $1 | tr 'A-Z' 'a-z'`;
    MAJOS=`echo "$(tr '[:lower:]' '[:upper:]' <<< ${OS:0:1})${OS:1}"`; # linux -> Linux
    # 1. Create img folder
    mkdir -p ~/workspace/amirr0r.github.io/assets/img/thm/$OS/$LILVM
    # 2. Copy screenshots into img folder
    cp $THM/$VM/img/*.png ~/workspace/amirr0r.github.io/assets/img/thm/$OS/$LILVM
    # 3. Create WU post
    prefix=`date --rfc-3339=date`;
    wu_file="$HOME/workspace/amirr0r.github.io/_posts/$prefix-thm-$LILVM.md";
    echo "---" > $wu_file;
    echo "title: TryHackMe - $VM" >> $wu_file;
    echo "date: `date --rfc-3339=seconds | cut -d'+' -f1` +0100" >> $wu_file;
    echo "categories: [TryHackMe walkthroughs, $MAJOS]" >> $wu_file;
    echo "tags: [thm-$OS, writeup, oscp-prep]" >> $wu_file;
    echo "image: img/$VM.png" >> $wu_file;
    echo "---" >> $wu_file;
    tail --lines=+2 $THM/$VM/README.md >> $wu_file;
    sed -i "s/img/\/assets\/img\/thm\/$OS\/$LILVM/g" $wu_file;
else
    echo "Usage: bash thm-migration.sh <VM_name> <OS:linux|windows|...>"
fi