import os

# 삭제할 폴더 목록
folders_to_delete = [
    "K-028495_json",
    "K-028450_json",
    "K-030542_json",
    "K-029672_json",
    "K-029646_json",
    "K-028466_json",
    "K-030283_json",
    "K-028442_json",
    "K-030409_json",
    "K-030615_json",
    "K-028482_json",
    "K-029726_json",
    "K-029356_json",
    "K-029617_json",
    "K-030555_json",
    "K-029616_json",
    "K-029683_json",
    "K-028762_json",
    "K-029357_json",
    "K-029734_json",
    "K-030188_json",
    "K-030347_json",
    "K-029464_json",
    "K-028740_json",
    "K-028716_json",
    "K-030286_json",
    "K-028441_json",
    "K-028510_json",
    "K-028684_json",
    "K-029580_json",
    "K-030422_json",
    "K-029647_json",
    "K-030474_json",
    "K-029671_json",
    "K-028555_json",
    "K-029514_json",
    "K-030365_json",
    "K-029419_json",
    "K-028602_json",
    "K-029727_json",
    "K-029429_json",
    "K-030157_json",
    "K-028493_json",
    "K-030411_json",
    "K-028463_json",
    "K-029677_json",
    "K-028422_json",
    "K-028469_json",
    "K-028683_json",
    "K-028408_json",
    "K-028516_json",
    "K-030199_json",
    "K-030600_json",
    "K-030636_json",
    "K-028792_json",
    "K-030158_json",
    "K-028592_json",
    "K-028476_json",
    "K-030520_json",
    "K-030488_json",
    "K-029643_json",
    "K-028477_json",
    "K-030285_json",
    "K-030528_json",
    "K-028509_json",
    "K-030173_json",
    "K-029687_json",
    "K-030378_json",
    "K-030478_json",
    "K-030244_json",
    "K-028814_json",
    "K-030225_json",
    "K-029644_json",
    "K-028763_json",
    "K-030616_json",
    "K-028743_json",
    "K-029513_json",
    "K-030562_json",
    "K-030649_json",
    "K-029642_json",
    "K-029675_json",
    "K-028475_json",
    "K-030356_json",
    "K-028790_json",
    "K-028635_json",
    "K-029676_json"
]


# 각 폴더를 삭제합니다.
for folder in folders_to_delete:
    try:
        # 폴더가 존재하는지 확인하고 삭제합니다.
        if os.path.exists(folder) and os.path.isdir(folder):
            os.rmdir(folder)
            print(f"{folder} 폴더를 삭제했습니다.")
        else:
            print(f"{folder} 폴더를 찾을 수 없습니다.")
    except Exception as e:
        print(f"{folder} 폴더 삭제 중 오류가 발생했습니다: {str(e)}")

print("모든 작업이 완료되었습니다.")
