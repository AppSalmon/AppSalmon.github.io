VM=$1;
OS=$2;
LEVEL=$3;

if [ ! -z $VM ] && [ ! -z $OS ] && [ ! -z $LEVEL ]; then
    LILVM=`echo $1 | tr 'A-Z' 'a-z'`; # Irked -> irked
    MAJOS=`echo "$(tr '[:lower:]' '[:upper:]' <<< ${OS:0:1})${OS:1}"`; # linux -> Linux
    MAJLEVEL=`echo "$(tr '[:lower:]' '[:upper:]' <<< ${LEVEL:0:1})${LEVEL:1}"`; # easy -> Easy
    # 1. Create img folder
    mkdir -p ~/workspace/amirr0r.github.io/assets/img/htb/machines/$OS/$LEVEL/$LILVM
    # 2. Copy screenshots into img folder
    cp $HTB/machines/$MAJOS/$VM/img/*.png ~/workspace/amirr0r.github.io/assets/img/htb/machines/$OS/$LEVEL/$LILVM
    cp $HTB/machines/$MAJOS/$VM/img/*.jpeg ~/workspace/amirr0r.github.io/assets/img/htb/machines/$OS/$LEVEL/$LILVM 2>/dev/null
    cp $HTB/machines/$MAJOS/$VM/img/*.jpg ~/workspace/amirr0r.github.io/assets/img/htb/machines/$OS/$LEVEL/$LILVM 2>/dev/null
    # 3. Create WU post
    prefix=`date --rfc-3339=date`;
    wu_file="$HOME/workspace/amirr0r.github.io/_posts/$prefix-htb-$LILVM.md";
    echo "---" > $wu_file;
    echo "title: HackTheBox - $VM" >> $wu_file;
    echo "date: `date --rfc-3339=seconds | cut -d'+' -f1` +0100" >> $wu_file;
    echo "categories: [Hackthebox walkthroughs, $MAJOS, $MAJLEVEL]" >> $wu_file;
    echo "tags: [htb-$OS-$LEVEL, writeup, oscp-prep]" >> $wu_file;
    echo "image: img/$VM.png" >> $wu_file;
    echo "---" >> $wu_file;
    tail --lines=+4 $HTB/machines/$MAJOS/$VM/README.md >> $wu_file;
    sed -i "s/img/\/assets\/img\/htb\/machines\/$OS\/$LEVEL\/$LILVM/g" $wu_file;
else
    echo "Usage: bash htb-migration.sh <VM_name> <OS:linux|windows> <LEVEL:easy|medium|hard|insane>"
fi