#map0 = affine_map<(d0, d1, d2, d3) -> (d0 * 1225 + d1 * 35 + d2 + d3 + 36)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3, d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d3)>
#map4 = affine_map<(d0, d1) -> (d0, d1)>
#map5 = affine_map<(d0, d1) -> (d1)>
#map6 = affine_map<(d0, d1) -> (d0)>
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 561 : i32}} {
  memref.global "private" constant @__constant_2xf32 : memref<2xf32> = dense<0.000000e+00>
  memref.global "private" constant @__constant_4xf32 : memref<4xf32> = dense<0.000000e+00>
  memref.global "private" constant @__constant_8xf32 : memref<8xf32> = dense<0.000000e+00>
  memref.global "private" constant @__constant_1xf32 : memref<1xf32> = dense<0.000000e+00>
  memref.global "private" constant @__constant_1x5x5x1xf32 : memref<1x5x5x1xf32> = dense<[[[[-0.144123852], [-0.203296721], [0.0245194137], [0.0424404144], [-5.772960e-02]], [[0.213269472], [-0.00469377637], [0.345114052], [0.136302114], [-0.259548783]], [[0.145365268], [0.112524837], [0.0500608087], [-0.0937015414], [-0.0550665557]], [[0.0901061296], [0.286698043], [0.111992478], [0.231037259], [-0.28824234]], [[-0.15258716], [-0.335655361], [0.156836212], [0.183970571], [0.124615282]]]]>
  memref.global "private" constant @__constant_1x256x8xf32 : memref<1x256x8xf32> = dense<"0x28DF883CAB19ACBDE456E13DA638AB3DBC37333D00FD153C49EA15BEE0A3E63DC09215BC86E30BBDC072EE3DF0EE283D5CCB19BED4EADC3DF221F2BDFA0A0E3E9138B2BD7CB5483D5209EABDAA9FD8BDE465163EB4CED43DCC0707BEFCFC413DF01A3EBCE451153E74D2F03DB644A43D7CF99A3D04CD7BBD50874ABC64BC5C3D294196BDF4B1433D666CAF3D80D8163E8F47AFBDBAF214BE00B68DB95562F8BD70BA103E3442DC3DCC01DD3DD07A09BC603AF03B08BD673D18DAD13CC69C8A3D39200ABE0178A4BDD2C2163E05F2DBBDD816F03D84D2A53DCEC495BD0EC215BE4A58B63D0065FEBA9C647E3D80F409BC04A3953D7C3A57BD503D97BCDAA5ACBDB060103D5E5AD7BD12A32FBDFC49163EE095D73DC2260CBE9785E5BD645572BD48FCA13C96EA01BEEC816ABD3085AA3D101AAF3C84899C3DF2D79C3D8F2A89BDB84B06BE4C6BEBBD68FAE5BD0A110CBEF77799BDD0D2DBBDEA4E4EBD347D6F3D7028153C3640893DDA2FB63D80BEADBD3C37D1BDA8D0003EB65EC6BDF6F6873D428A61BD084C833D48E2BCBD573E11BEB421423DE0E3FE3C8C91B33D4A0A003E80F6433C7AE8033EB8DFEDBDEC45EF3D883C213D84357D3D081318BE2A24A3BD28D6E93D756DC3BDD2B3F1BDAC000D3D90C8FE3DDA97DEBD6D2BA7BD14CDEC3D26203EBD9F77B4BDDC41EDBD74AE1D3D42A2B9BD66F2AC3DCCD0F9BD4079A33C706EDD3DEB25F0BD5EA1ACBD4421E43D58A2123D909905BD2410773DA8C56E3DB0DE99BDEA648FBDFF1AE1BDBA9A08BEF0F9C83D4B5D8CBDA2D393BD1CFEEC3D5CFC2C3D78D908BDDC5CF43D68B2B8BC549B30BDC5D4FFBDAA84143E169316BE14A7E53D105C223DA586DCBD2896DEBD7622143EC76B08BEA2770A3E8E8B4EBD668F00BD2E1DC43D2B919DBD02D0903D2060093CA8D18BBD6878803CC4FDE6BCD655133ED881A3BC003D853D6C0DA4BDF0B195BC84FC023E0005AB3B7CBF093ED0AF783D302216BEFD7286BD70E8873C76025FBD484494BCD2A45FBD00AFBCBB0018D43C88DA293D80CBDABBF609BC3D200E7EBCE8102F3D2A65AFBDEA6708BD524915BD4406133DE096ADBB5F7D18BEF19A14BEC4EA063EDABB153E7C5BD23D90B9CB3DB84AC43C34D5013E700C76BDE2F0F4BD3A6A79BD8A6AFDBD7037953CD04EFD3D16729B3D4429FE3D1857F4BD809625BDCE83A53D4621ECBD30E91B3CC0F535BBBE7E973DC4ADA2BD2C9F153D8A7BB13D72340F3ECAD7DBBD42CC67BD0EEE02BE7279053EAA1E10BE8C1715BE0CAFF5BC6A940BBD6225C5BD3BC50DBEC28003BEE89AEA3DD1FF15BEB070863CD4E00DBEE4FED43D9CFE2DBDF85B883D67E5D6BD4EF02CBDCC15123E0208A93D1069CFBCE073C93D347E6C3D003D79BA7067043E3670B33D6710DFBD1C00043E4206BE3D88B7A63C5C790C3ED0B8213CFB26E0BD627C25BD50096C3CA033BCBCB4B0C73D28A3EC3D009CE63D68979DBD811507BEB063853D724200BEA0F6143D83A3C7BD88510BBE083E17BEAF79FABDB8A7113DE401A5BD8B689EBD9246F7BDE2ECB0BDD8E0AB3C9A3C2CBDE07000BE10BBA73C804EBA3AB0AE043E5098D43D8972C7BDE09ABD3CA034143EA435D83D655996BDD98DFABDA49B08BEC646B03D04C3BA3DD0858ABCDCB815BE15F414BEE8F5313D3055073C24C008BD82F8A43D7046023DC45B193E83D290BD705B2F3C64EF14BE40E6E63C805EB53A685C573DC0A6073BF07578BC9C2A22BD004D5C3A90E944BC6434D73DDC54293D2C7B133EE8210CBEE4D26EBDC825183EA22505BEB0BE113C1C58263D20CE063EB8722F3DF20DD1BD6C3E0C3D72CCACBD4001C03DF047073D74F7943D562351BDDA1432BDB62536BD58A347BDA0C6963B9E9DA43D8887A13D40FE27BBD46CADBD3CD8813D765148BD0418163E00046D3C18992CBD48169EBC96628B3DD0833A3C5CCD0B3EA8D215BE9BE386BD4081DA3BEC620DBEB0CB0E3EDC5AD9BCD31E03BEEAC7023E3856E83DD01B90BC085BEABDE225A53D502B3A3C2098873BC060403D820A9D3D105333BC9ADC153E1405E13D7DE6ACBD7067F93C20F546BCC04445BB480AA4BC20614DBC20D61EBCD2DA043E6666BCBD2F5388BD3606D7BDC87C95BD4C70193D70EA103C64DF803DE4CF2ABDC893CBBCE49FFC3D269161BD5E05BE3D7098803D34F44D3DD06DA43D361FC1BDF9F0E9BDE835F13CF02A123C746E7B3DC01205BD5433173E0037FBBA475ACEBDAE7F55BD904A0B3E807300BB1879BFBD04349ABD80A0C33CACB4413D0AB365BD20F0063D3096763D50321D3D2EFC42BD4B8EF7BDD1BAA7BDF48B5F3D7EDF2ABD006C7CBA9C8F51BD701C6EBD2853F33DCC7459BD2069BABBA01004BC68729B3DB85281BDDC33FF3D5871193DB06250BC388C703DB0E882BC4016243C266471BDE08BFF3BED0001BE50FF3E3C4EDE843DF42AD63DA08736BD9A33163EA0F915BE2C93E43DA063B33D1CD9ACBDF059103E524AAA3D12B9D1BD9E02043E4C1D8D3D6894DF3D20749ABC0058C7BD20CD073D18E6E6BC485639BD64AF4E3DC2338BBD15FE06BEA0DBBEBC5214003E1AB9023E08F1F33CF4BD00BEF2C2A03DEC29E73D8D84D1BDE3A3B3BD40412BBCAC46FA3D1018ECBCC0619ABBC568D8BD448382BDD24A44BDA471943D14CC01BEA04B61BCB080AB3D483FB0BC4E704CBDB652E5BD882B1F3D6BE3C0BD94E02D3DA8B9D2BD50B1CD3CAC275EBDB88F963D5BF30EBEB3FDD9BD50FFAE3C50D0C7BCF039DD3C206FCF3DB2120FBE52E3083E692B02BECCB741BD9CD006BDBA72EEBDD56885BD486CE73D9848D23C346F06BEC20816BECE4E8DBD0C5ADF3D47FF8ABD7A879A3D3A2E48BD80C7ACBC9C3C0CBE7AAB803D692C11BE466B8ABD2E0522BDA4F60A3DA81691BD5E66873DECBB393D5B7CD2BDA82E17BE5069FA3D80CA2BBBF81BCDBC3C9B763D6DF30BBE10AB0DBD368A003E26FBB03D30483FBC56AFBA3DEC390DBD48AA163D4EF119BDF896923DF916F8BD20A9F23D2CA6493DAA90F4BDD4F55CBD5C95023DEC83E1BDC56816BEEAA3053EA2771EBDDCB1783D24C6E03D53CEAABD48A633BD3C7567BD00F34E3A0091D13DBC8226BDC0460FBBFE73073E207DC03C4C6EAD3DB626CABD4EC6063E6854AB3C840BF7BCCC16063DEF5C9EBDD8B616BE18DFCF3D282EC23D04ACE83DAC3EEF3DFC1EDE3DD1F695BDD431B73D2A30843D124AF3BDB23D59BD605E073EE619133EB497913D30F767BDA2A9083EF2EEE3BDA48A463D88D842BD8424F93D68820BBE228D9B3D998087BD221BC73D20EB8E3C1C65B9BD2029483C706F85BC6F4883BDF4C1153E2CEA03BD2069D43D20BDD13D5B5895BD004F0C3B2E230F3EFC0314BD6EA5023E869FFBBD08B9FCBC96C8FBBD90B3333CE0BB883C00611FBA06268DBD68F7533D309F453C8CA7A43DAAD7003E54D5E13DCC340E3E6439CDBD8690CCBD00C7143EFEBDB53D54F224BD2607DFBD9477E83D98F7CABCC0694E3B2CFE6B3DA6A98ABDCA22A23D401C0BBCBC36C4BD20AE003EC59DD3BD247831BD00BC55BA77DA06BED0F1353DFCA0FCBC94A09A3DD444AE3DE8602C3D409FC4BBAE4F30BD2B8FAEBD5E006BBD4E8413BE74FF0DBEC4E4DF3DF09356BDA825A33D806A8ABBE8CBF03D28DEDC3D40B29DBC3B64D0BDFCCF1B3D6C89203DB83B003D78DDF63D1CBF92BD2021373C3088783C894819BE68989D3D689F703D47E28CBD50B0E33C401AA93C70DE5B3C01BDBBBD061D5DBD009BA43DA8A200BD393AF4BD9BFBCFBD46DABFBD44DF953DD529ABBD00B620BCBC38A93D38AB353D9CB88ABDD8E1E2BDE831323D904010BE32D3133ECAB5C83DA099523DE03A9EBBE039DC3C662016BE5E3D0B3EF2C5F0BDAC93D4BD503D7FBDCAFBBB3DF3F68BBDEE9022BD3435693D803EE03A36DC74BD9601913D186D033E2F1AB0BD30F0473D414BF2BDE008FF3C002B83BDE39312BEDFFDAABD74A4F93D719A9EBDE2026EBD0855BBBC8828D93DC832073E970A0EBE06950F3E559E06BE6C72E0BC10F95F3C7538EEBD60DDE4BB0053C03B1552B2BDF2A2A1BDC887D1BC14F3CA3D74EB3D3DC0240D3E309C6F3D84B571BD819D0CBEA8126DBDD0E4FB3D38DBC43C6685D2BDEC73E13DD822F7BDC08866BB9748C9BDEFF0FCBDF62E04BD60A3493C20A55CBC34950FBEC422993DEC8F0F3E6878133ED831973CC8088FBD0044A63BC006993BEB1B0EBEA897693D10A59EBC88C398BCA872FF3C8840D23DDA2985BDD818923D9C9BC33D24B1E43D9C5AA5BDC034F63D2CE0BDBD68F9A63C7CF0433DD566A1BDC0AF02BB1B3989BDA8F0173DA6C4C13D2EDD06BE2827BC3D9C1A843D8278C8BD8E2D12BD50CC06BE4F4C9DBD965D78BD6287083ECEEDB9BD6E1FC6BD923450BD27FBA9BD601AF33B8894C8BD50743F3CA0D887BC00AC323C1068F23D8765A8BD523C0EBDE0D400BE8AEF923D40F3923D0C05313D80690E3E8872ED3D92E4163EA8A7D0BD0CB6DF3D6056F73CB000B83D7827CABC30BCE7BDBC84E53D8A9BC53DC48FF5BC6C82EFBD895BFBBDE0C792BB315FDFBD7081CC3C11E20ABEA0EA5D3DA4D8F23D1C8037BD046EB43D307CC8BDECE0ED3D2461FE3D4BF783BDFAFF8D3D547F133D3FE9AFBDE84EFD3D52EE043E4CAADD3D67A186BD6291093E947290BD985EEE3CA45F5EBD98E0093E16DCD3BDE040FA3B7C85793DE09AC73C08F24C3D3F7319BEC0CF6F3D206EDABDF8D2243DA807253D849F103DC03A9B3CD46E093D28E4EA3C04F7FA3DBAAD01BEB0F1F33D0DF806BE3A51D6BD40D656BC0293F9BDCA09A7BD0A9C093E6E0D153EF4ADFA3D009F703D488A8DBC9F848BBD4099C23D5A73F1BDC9A9E3BD14470EBE9197EEBDFFC60EBE59319FBD5A9D983D1C33EE3DA4F974BD56B414BD46B3F2BD202A7D3D38A5A73D4C4014BE685EA4BC78CA013EB4881E3D7A50B1BDF047C7BC765423BD04E1D63DA06101BD543AAF3D3ACE023EA0272B3D0A61BABD1443E33D64E5773D8A817ABD08D1F53CA06C23BD780C10BE2055CDBBA64C5EBDD7D3B0BD00EDEC3C004C05BC80600F3EC0CCE53B7B5A93BD90EDD93C9C03DD3DCEE986BD2029D03BF9980CBE7C97EBBDBE38BFBD8C50913D0EC6903D0035013AE4B804BEDCA5383D303B2DBC260DFFBDFC710C3DFFE1EBBD80DDCF3DB0D7E33DBA2F29BD33D7BCBDD2D7A73D1044EDBD749CE53D36C913BEC2974ABD584C72BDA88CB53DA6CFC9BD3CB3FBBCA4D212BE7E9911BEA0EB6ABC42D7033EB064FB3CA89B6B3D58ADA8BD00AE673D00773FBB638085BD686C153E5E1345BDCCD6943D07C606BEC8E3103D554B8BBDD0D6F9BD76F318BE4469ED3DEE1012BEB417D73DCEF8A43DA833AF3DA839983C308BB5BD6236B53D1EB4C9BDACC6D8BD40A5C2BBE87609BEFC1ED33DA063B4BC60A2303D3011C93DFEB213BD4C5C183DB04C00BEC4702E3D8F1F8CBDB08E83BDC0D701BE6834C93D22C4133E28378F3D60DAFABDB814B0BD4C6CECBD2C239D3DF83559BDE01885BB3E8F9C3D63EA14BEE346EBBD30DF0CBEB667A63DAB2DB0BD4893E93CF250FBBD7693183E827E4CBD4AD42FBD14B7E9BC5C83103E3AA0E0BD009C0DBB449C6A3D606B353D0FB5F8BD59FF03BEA79795BD00BF053ED84802BD8056113C50736E3D36CA0F3E6058203C80B6883B3CE64E3DD099C2BDE6A807BE1C41F7BC1A98F1BDFE7BD8BD8A8AECBDB081953C625F19BE3B9BA6BDD49A3EBD40E1DF3C5059AFBC8C83113E589D0CBEB46DF9BC580774BD88B15A3DEBCBA3BD0A1A09BE206B9B3C4D87F6BD7B1987BD80E7613D70B7E73DF0065FBCC039A53CAC4F333DA05ED93CD4EA43BDDEC6DCBDECC0A7BD98DF093E70EE12BE004764BB347F083EEA6AD6BD90AB64BD5878D53D60378C3DC011E73D286DD13CBC952ABD6057933CBEAD11BE24204C3DBC3555BD2C4C253DF02BF33DC69CC03D4CDC1ABD78A1E23DC00D06BE2057EEBC60CA02BCAB3019BE4F8D07BE0A3260BDA05D3A3C42DA073E58E0F13C600DF03C894AE7BD00748C3AF491143EBA121FBDA0CFE2BD6887283DB2A517BE67CBBCBD28C20ABE5C2C1F3DC50298BD2C14C13D56A2103E76EBB63DF459AFBDD4160ABE002460BC809664BBEC42703DE08F403C6167A2BD68AE653DA6208A3DD65A4BBD6860763D340E943DC08F03BCFCD9113E0014733C41BCABBDEC4C09BEE34CF8BD40ABB8BCC06F18BD600EE93CA839BABDF024FA3C045E583DBC768D3DCE93B13DAC6D023DFABEA63D5E4F11BECCB1D53D68ED88BD14B0A9BD3A2217BE2084E43C213789BD6084EBBB5E05BE3D64F39E3D419AF4BD42A80B3E2442E3BDD24B813D002531BB782181BC3B6FA8BDDC47FE3DE0A4433D42C100BE202A0A3C6496013E9001703D32CE09BE2E769BBD880889BC02BB9F3D8E970E3EF4B94E3D969F03BD32BBAA3D782D033D909315BE9C3FC4BDB9A981BD4AB5A4BDD009933C9676963D61F302BE8AF64FBD7849E13D7C71E7BD9CAD84BDD8CAE5BDDE8CF2BD7034F83D09CAC0BD0072F9BB58E7A6BD00C283BCD4EE06BEF0600F3CB7D0D8BDB847A23D7432A53D2009A6BC43489EBDD44D473D40AEEDBB1CE6193E18D02B3DE459A53D350099BD9003BF3D521500BEF0F94DBCC442003EDE56EBBD00600EBB52B68FBD1C89523DCF3413BE5457E33D30EA0ABE48C208BEE82B1A3D0C3A643D4856D83D10F329BCC679E6BD087BC43DB6C1933DF170B8BDE63F043E9771C6BD90D1CCBCECCD4CBD17BD91BDAE4E72BDC0B5EE3D682C9E3D285ADD3D7463B03DE8EFFC3DAD0E03BE18BCF03DBC460B3EC639133E5867173E28D6C43D86E11FBD98A88C3C53CFE3BD244BE23D44550EBD6E939D3D48290B3E20F964BD68A019BEA004113C92C101BE93B3AFBDCE9343BDC4C01F3D2815D8BDF81CF03D42CCAB3D10DC113C322619BD769E0CBD50F3B43C3E4E19BEA01F78BDA8F6E83D686C3F3D30A8FA3D9C7AE6BCEAF68D3D28C7EA3D801A473BBF0E13BE80AF853C9C11E1BC80A60D3ED086673CC0AB83BB260DE2BDBAA2903D94D9A2BD169E85BDD446273D9024F1BC00AD75BCFA7BA93D00359F3ABCAE13BDA418113E5913CFBDF8D896BCD24F153E4C103DBD06407BBD52F477BD209816BEF83AD0BC8072133E568E143EF79297BDF4AA163D70DF813CB0E83FBC60F811BE3455D43D748B493D0C5C3B3D686F98BC2BE197BD9E7AC7BDF8DCEF3D6DAAB3BDF00BBA3CC403783D7CB7CEBDEC85F5BD10D588BD005F073AEAB962BD44DCFD3D18BEDD3D22C8153E9279AD3DF7D391BD7CF2123EDCBE0F3D047858BD7A2C5BBDB8196C3D109CA3BC5491123E302626BC72060FBE084D173EA424163E10B568BDD075023D9C8D27BD58D1B1BC60857FBDE0F4B3BBF00A68BDF404733DF023493C137CF9BD4A100C3ED862F43C287C80BDC840FDBD3CDB97BDA951FCBDF0ACCA3D085077BD30A03F3C40632EBC8ED0C83D207674BCB87811BD95F817BE0852963C9CDCE4BDDFFE01BE0664843D809B97BADCA997BD6E5E01BED0610F3E005D003D0DABBEBDC045E5BBED908BBD388D833DDCF9B6BDBD10CABDC6DD9D3DFD4BAFBDD7E10ABE22E00C3E5839E73DA45B633DE86D97BC9EB0913DA8D2073E5BF213BE6805063E08C3913C3CD60A3DCA8461BD64288A3D4087BCBBC07E303B16C092BD58F30B3EC8248F3D86E376BD50213C3C4C4A73BD7494B53D8E66AC3D1BC1CBBDA098883BC757EEBD30F3B63C88CCBFBC6223A83D625FD8BDD005B3BCB4623C3D40E3133C028100BE56B603BEBAC89F3DBA2D0F3E48D18E3C2231C0BD622B863D167305BD0801BB3D081F123E4050283C6E6270BD34274B3D4472163D0CA6D5BD78D2D83D2E56C9BDF29ED6BD30ED013DA60BC93DB05A7F3CD87BB33C0A16FDBD6043F13BF81DAEBD5070A23D202A073DDF5B0DBE7814B1BC4C98723DCCE8B2BD33318EBD3C4607BE04EAE1BC85F709BE92FB47BDB09EE73D7EDB8A3D7A3B0FBD0E1EA0BD78FF843D6B8D01BEA227033E6098FE3DD44A22BD26310FBE450809BE808A8A3BFC39ECBC907339BD2424983DE07D283DC2B324BD8053E23DF021E13DA06585BC3EAAD1BD684AB93D4EAEBABDAA4D9CBD59EC00BEBA14C73D0C84383DCF1C84BD6360A1BD448718BD08CB623DDD3BE2BD80A8673D18C6903D00BAEA3B0BB1D1BD088D183D260D943D5863083EB3F113BEE1FD17BEC624943D4AA2ACBD7884A8BDECF1B53D8030F1BAC22AA2BD80F0C1BD9A9004BEB763D6BD644E083DC045EB3C6C5F7F3D624B163E665E55BD4CA4443D188A8F3C9570E7BDA6214EBD3442C93D3C6DDA3D57AC92BD7246093E8FBA12BEAC1DF3BDB8C4B43D98FBFB3C8E5CDFBD51EA02BE20AFF53C44481E3DD4D4783D4827403D5874CCBC70FDE33C74E1FE3D582B7F3D848652BD70535A3C61EB02BECBDC10BE581B023EE6AD003EAA78A33D3AF402BEDEBB053E5EB208BE1F920ABE407AF33B6EB608BEB00A223D572A16BE40FD3EBCCD3208BE9265C63DEAE9DBBDD02B3FBD48DD91BCA47EF03D9AE752BDA825193E14FC513DC86CBB3DB230BF3D3A2F0B3E406D5A3C54F20D3E8095C8BC3EFD19BE2801F43C88DD6BBD30F76BBC4C4509BD9484FD3DB8FE753D0E05883DD875CD3D603ABDBD90660E3E4B091ABEA86EB6BDF01C3B3C4C82583DFECAA23D9AAD923D4C1C02BEDC8A163E5BB705BEE4E7163D781DECBD0275AB3D9C59363D8FA2D5BDC8B4143E0DFBFABD5FC382BDF8E0B0BC16A4003E80DF133CF664193EB803F83C983C7D3D9043023D84B2703D822C123E4083A13D408E84BB3896F23D6E85943D0894E0BD9479043ECCB0123E767AC33DD61226BD6C52043D498EA7BD8A06193E0296F1BD7EE075BD948C573DD0AD983CA438B43DCC10A63D7C68B03D4824C7BC952D84BDA0DEAABBF20155BD2624A63DEE1A0E3E3468CB3D68FD743D4EAB0B3E1C55F93D85DA01BE124BE2BDA0A3B6BBF03E02BCCCB918BE508C4F3CFA55C2BD1080403D007C7F3CB658133E54DD053D0FEEA1BD5897E13D6072DC3DE0CDEF3BA26B74BD748232BD867BBD3DE2FB80BD0393D5BD20EF31BCAA3011BD00F493BA6ECE92BD121C153EF471053DA8C2643D246DF23D62024BBD54E27E3D201FE33DA8EB283DC63AB43DC8789CBD4865C5BD5C3ECBBDFC95E83DE8C2B33DFA32B5BD10450F3CB064343CFA4A083EC030283C049D563DE843BBBDB0A84EBD4CE63DBD8CE743BD64B616BE18834D3D5CB3183D34EFE73D447DE53D6884C33D2822F93C846F07BE313DCABD0078C3BB00739C3B90FB6CBC3E2AA83D82BA113E1F048CBDB85E7C3D36B99CBDD4E63FBD1A9F053E034CFEBDC60CEEBDC0682BBB38E2BA3D5C7B023EA4CFF53D4E62053E007E953AFBCC00BEA4AEB03D00F8B538C0A9A33D33C4D6BD004E6C3BE049C13DF03F273D14BBC23DBB9DC0BD742615BDE8A994BC4293923D38011E3D5814123D6C85913D46F7AA3D8614033EACB0FBBC2099B3BBB83EF53CD189C2BD9B5BE6BDAAE0A2BDA0DD053C8C03FB3D206C263D4A7103BE3B5C0DBE3EADA63D100B90BC6E4B063ED29E8FBD5188AABDF4CCDBBC0A81D6BD4AA617BE464112BEA8F71A3DB08B653DA59891BD9A12133E34AC663D48528A3D1040C53CC05DA7BC005694BC4078B73B3DDEAFBDAE7E923D6F4AB4BDDABE8B3DC3E58BBD80D1FD3C382CC63C5F579CBDABE7EBBD7077FD3CB07863BCDCEE17BEA64A163E7474BC3D307FC13D323DBA3D00AD163EFC5ECCBD0D12F6BD2E43AB3D2EFCDDBD03EEE8BDCAEB033EEA5AF1BD2836083E0078D7BC94E9993D209AD03DE837D8BCB7090BBEDAF6143E4AAACEBD1E650A3E98DC86BDF6DD14BE3884A1BC55E9A1BD4890613D9CAF583D5C60FE3D9CE4263D889DC3BD08A95BBD106D04BC1E54CFBDD2845DBD4D1810BE78791D3D20C217BE0CE3953D84680E3E79CDD7BDF4121ABEDC55E6BD48AA083EC0D065BBF70F03BE3C0DF3BCC851D1BD5E475CBD58149D3D60D4893CF1A4F5BDAE3DC93D52B1BD3D60D19CBBBE420E3EB4FF41BD104A03BEF0B0383DB43331BDCAF2AABD9D8F85BD48AEC33D28F526BD783FD3BDE89F0F3E88B9F93C8461D13D7671AC3DAA3D2EBDA489F73D804109BC568FB7BDE87B5EBD7D6BACBD5179E2BD761D873D47A40ABEA0B5E5BBA4ACD43D80EDA63C30EA333DB513F5BDD0B4A1BD9003223DB895C73D94BD233D26599E3DE8A1F53D3203953D484D363DAEA38E3D0E3BAABDAC20FE3DC0519DBBE6520ABE2859B9BCEC568B3DA285183E4F9516BEF055B73CF8C3BDBC86BD023E3CB7B9BD000C66B9B8EC023D0C84F23D1844023E577302BEF59708BE38FB103E60E636BD422FE4BD24CC133EC2AC9C3D1CEF753DA4AFCF3DB6270C3EC619BDBD986522BD74B0213D6871B6BDCC810E3E7CEEF2BCFEE381BDACFCDEBD483E9A3D322D023EA978A2BDD052793C182BDD3D181BD53CE093CC3B38608F3DBECE13BE8887B7BC7886FF3CDA779A3D867984BDB1E408BE2033DC3B80700DBC9EAA18BEF39C0BBEF8FE563D00607E3D72E9EDBD5032043E65C514BE44EA0BBD64AAAC3D00DCF6395841E0BC1B258FBD6E03C7BD84CD233DCA8273BD5EE1F1BD4858FF3D504BF53D1029F23C20B09B3D22370C3EC8E9163E1C36003D9812FD3DE8DDE0BCD0376EBC1B1297BD54D660BDC3F5CEBDC36406BE3C1A233D5CBC1B3D3F7E9BBD14EEC73D4ADFA43DF0E1743C286815BE907A68BCB48E01BD9829BCBCAD4D8FBD082FED3D44BFF43D045A303DD61F933D0821DA3C0000AAB67C0FE53D484D343D70DA0E3EBC59213DD2CF923D18E1FE3CF4A7753D4047F23B48CFC33DE08C9DBB47F58EBDB64926BDF858E33DB40BF9BD183B85BD46AAA13DBE382DBD7B0E89BDD83C923CC4BB313DE421CD3DF62C063E8C46D73D08FAD23C5CE9C23DCC04663DBA235BBDF8CDFDBCA072F5BB744A98BD42E597BDD64705BE00994B3B2D1E12BE6C91973D1EE9F3BD603C30BC56659C3D6428A5BD22ED4EBDA034463D2646013E20DBAE3CA028DA3D33D8A1BD128510BE10D52E3DB079BFBC09239BBDC0F3E23B4BB191BD7023133D685804BED4D0D23D5ADFBA3D18E5F63D28714FBD18A63A3DDA88C83DF57593BDA681EEBD4039F9BB6137FEBD8071473CC4E7DBBDA89FAA3C90C03A3C2415FC3D7096A63CFCA0963D70B4143EA86746BD4EF9F5BDCEFE0BBDA0DB203D54AD83BD801EF7BBB7AC0DBE60C486BCC880843DD8B1193EA07EB7BC8A945FBD90B9CF3C683BAE3D5A1ABABD60B6A13D6041923C904B163EACC7DBBC60D2A2BC0409123EB0E0413DE94A16BEB40F103D3A2E0C3E45E303BE429007BEFC28753DCCA1B33D783A08BEF712E8BD46B827BD96AFB13D746C533D2C8A063D78D6093E643CDC3DE03C46BC4014843C1D9CB8BD797600BE9C02C33D7EF757BD6828D6BCC47A01BDA441E53DD0118EBC640A1DBDD4B2163DA07AE13D3CF37D3D3AE490BD0E44BABD60DC2C3CC66E03BEAAD079BD8DF983BDB65F103E88F4B3BC4082743B34900C3E08EED2BC1AD2DDBD04E8E83D208EDD3CC0B129BB">
  memref.global "private" constant @__constant_1x8x4xf32 : memref<1x8x4xf32> = dense<[[[-0.431032777, 0.339656174, -0.0172508955, 0.530783117], [-0.131361604, -0.686532557, 0.29633683, -0.190197289], [0.114347279, 0.56687206, -0.5206815, 0.0635936856], [0.278496087, -0.701272607, -0.192214727, 0.0925094485], [-0.263988972, 0.210848868, 0.702368438, -0.632420897], [0.552770555, -0.32865116, -0.0839602351, 0.335239589], [-0.466998219, -0.367347717, 0.669422566, 0.683026611], [-0.653910696, -0.016515851, 0.00850236415, 0.640155136]]]>
  memref.global "private" constant @__constant_1x4x2xf32 : memref<1x4x2xf32> = dense<[[[-0.649729251, -0.36405158], [0.711094856, -0.165482044], [-0.528160095, -0.548771381], [0.198056698, 0.657765388]]]>
  func @main(%arg0: memref<4x32x32x1xf32>, %arg1: memref<4x2xf32>) attributes {tf.entry_function = {control_outputs = "", inputs = "x1", outputs = "Identity"}} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 3.40282347E+38 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<4x35x35x1xf32>
    %1 = memref.get_global @__constant_2xf32 : memref<2xf32>
    %2 = memref.get_global @__constant_4xf32 : memref<4xf32>
    %3 = memref.get_global @__constant_8xf32 : memref<8xf32>
    %4 = memref.get_global @__constant_1xf32 : memref<1xf32>
    %5 = memref.get_global @__constant_1x5x5x1xf32 : memref<1x5x5x1xf32>
    %6 = memref.get_global @__constant_1x256x8xf32 : memref<1x256x8xf32>
    %7 = memref.get_global @__constant_1x8x4xf32 : memref<1x8x4xf32>
    %8 = memref.get_global @__constant_1x4x2xf32 : memref<1x4x2xf32>
    %9 = memref.alloc() : memref<4x35x35x1xf32>
    linalg.fill(%cst, %9) : f32, memref<4x35x35x1xf32> 
    memref.copy %9, %0 : memref<4x35x35x1xf32> to memref<4x35x35x1xf32>
    memref.dealloc %9 : memref<4x35x35x1xf32>
    %10 = memref.subview %0[0, 1, 1, 0] [4, 32, 32, 1] [1, 1, 1, 1] : memref<4x35x35x1xf32> to memref<4x32x32x1xf32, #map0>
    memref.copy %arg0, %10 : memref<4x32x32x1xf32> to memref<4x32x32x1xf32, #map0>
    %11 = memref.alloc() : memref<5x5x1x1xf32>
    linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%5 : memref<1x5x5x1xf32>) outs(%11 : memref<5x5x1x1xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
    }
    %12 = memref.alloc() : memref<4x16x16x1xf32>
    linalg.fill(%cst, %12) : f32, memref<4x16x16x1xf32> 
    %13 = memref.alloc() : memref<4x16x16x1xf32>
    memref.copy %12, %13 : memref<4x16x16x1xf32> to memref<4x16x16x1xf32>
    memref.dealloc %12 : memref<4x16x16x1xf32>
    linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%0, %11 : memref<4x35x35x1xf32>, memref<5x5x1x1xf32>) outs(%13 : memref<4x16x16x1xf32>)
    memref.dealloc %11 : memref<5x5x1x1xf32>
    memref.dealloc %0 : memref<4x35x35x1xf32>
    %14 = memref.alloc() : memref<4x16x16x1xf32>
    linalg.generic {indexing_maps = [#map3, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4, %13 : memref<1xf32>, memref<4x16x16x1xf32>) outs(%14 : memref<4x16x16x1xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %41 = arith.addf %arg2, %arg3 : f32
      linalg.yield %41 : f32
    }
    memref.dealloc %13 : memref<4x16x16x1xf32>
    %15 = memref.alloc() : memref<4x16x16x1xf32>
    linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14 : memref<4x16x16x1xf32>) outs(%15 : memref<4x16x16x1xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      %41 = arith.cmpf olt, %arg2, %cst : f32
      %42 = arith.select %41, %cst, %arg2 : f32
      %43 = arith.cmpf olt, %cst_0, %arg2 : f32
      %44 = arith.select %43, %cst_0, %42 : f32
      linalg.yield %44 : f32
    }
    memref.dealloc %14 : memref<4x16x16x1xf32>
    %16 = memref.collapse_shape %15 [[0], [1, 2, 3]] : memref<4x16x16x1xf32> into memref<4x256xf32>
    %17 = memref.expand_shape %16 [[0, 1], [2]] : memref<4x256xf32> into memref<1x4x256xf32>
    %18 = memref.alloc() : memref<1x4x8xf32>
    linalg.fill(%cst, %18) : f32, memref<1x4x8xf32> 
    %19 = memref.alloc() : memref<1x4x8xf32>
    memref.copy %18, %19 : memref<1x4x8xf32> to memref<1x4x8xf32>
    memref.dealloc %18 : memref<1x4x8xf32>
    linalg.batch_matmul ins(%17, %6 : memref<1x4x256xf32>, memref<1x256x8xf32>) outs(%19 : memref<1x4x8xf32>)
    memref.dealloc %15 : memref<4x16x16x1xf32>
    %20 = memref.collapse_shape %19 [[0, 1], [2]] : memref<1x4x8xf32> into memref<4x8xf32>
    %21 = memref.alloc() : memref<4x8xf32>
    linalg.generic {indexing_maps = [#map4, #map5, #map4], iterator_types = ["parallel", "parallel"]} ins(%20, %3 : memref<4x8xf32>, memref<8xf32>) outs(%21 : memref<4x8xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %41 = arith.addf %arg2, %arg3 : f32
      linalg.yield %41 : f32
    }
    memref.dealloc %19 : memref<1x4x8xf32>
    %22 = memref.alloc() : memref<4x8xf32>
    linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%21 : memref<4x8xf32>) outs(%22 : memref<4x8xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      %41 = arith.cmpf olt, %arg2, %cst : f32
      %42 = arith.select %41, %cst, %arg2 : f32
      %43 = arith.cmpf olt, %cst_0, %arg2 : f32
      %44 = arith.select %43, %cst_0, %42 : f32
      linalg.yield %44 : f32
    }
    memref.dealloc %21 : memref<4x8xf32>
    %23 = memref.expand_shape %22 [[0, 1], [2]] : memref<4x8xf32> into memref<1x4x8xf32>
    %24 = memref.alloc() : memref<1x4x4xf32>
    linalg.fill(%cst, %24) : f32, memref<1x4x4xf32> 
    %25 = memref.alloc() : memref<1x4x4xf32>
    memref.copy %24, %25 : memref<1x4x4xf32> to memref<1x4x4xf32>
    memref.dealloc %24 : memref<1x4x4xf32>
    linalg.batch_matmul ins(%23, %7 : memref<1x4x8xf32>, memref<1x8x4xf32>) outs(%25 : memref<1x4x4xf32>)
    memref.dealloc %22 : memref<4x8xf32>
    %26 = memref.collapse_shape %25 [[0, 1], [2]] : memref<1x4x4xf32> into memref<4x4xf32>
    %27 = memref.alloc() : memref<4x4xf32>
    linalg.generic {indexing_maps = [#map4, #map5, #map4], iterator_types = ["parallel", "parallel"]} ins(%26, %2 : memref<4x4xf32>, memref<4xf32>) outs(%27 : memref<4x4xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %41 = arith.addf %arg2, %arg3 : f32
      linalg.yield %41 : f32
    }
    memref.dealloc %25 : memref<1x4x4xf32>
    %28 = memref.alloc() : memref<4x4xf32>
    linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%27 : memref<4x4xf32>) outs(%28 : memref<4x4xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      %41 = arith.cmpf olt, %arg2, %cst : f32
      %42 = arith.select %41, %cst, %arg2 : f32
      %43 = arith.cmpf olt, %cst_0, %arg2 : f32
      %44 = arith.select %43, %cst_0, %42 : f32
      linalg.yield %44 : f32
    }
    memref.dealloc %27 : memref<4x4xf32>
    %29 = memref.expand_shape %28 [[0, 1], [2]] : memref<4x4xf32> into memref<1x4x4xf32>
    %30 = memref.alloc() : memref<1x4x2xf32>
    linalg.fill(%cst, %30) : f32, memref<1x4x2xf32> 
    %31 = memref.alloc() : memref<1x4x2xf32>
    memref.copy %30, %31 : memref<1x4x2xf32> to memref<1x4x2xf32>
    memref.dealloc %30 : memref<1x4x2xf32>
    linalg.batch_matmul ins(%29, %8 : memref<1x4x4xf32>, memref<1x4x2xf32>) outs(%31 : memref<1x4x2xf32>)
    memref.dealloc %28 : memref<4x4xf32>
    %32 = memref.collapse_shape %31 [[0, 1], [2]] : memref<1x4x2xf32> into memref<4x2xf32>
    %33 = memref.alloc() : memref<4x2xf32>
    linalg.generic {indexing_maps = [#map4, #map5, #map4], iterator_types = ["parallel", "parallel"]} ins(%32, %1 : memref<4x2xf32>, memref<2xf32>) outs(%33 : memref<4x2xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %41 = arith.addf %arg2, %arg3 : f32
      linalg.yield %41 : f32
    }
    memref.dealloc %31 : memref<1x4x2xf32>
    %34 = memref.alloc() : memref<4x2xf32>
    linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%33 : memref<4x2xf32>) outs(%34 : memref<4x2xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      %41 = math.exp %arg2 : f32
      linalg.yield %41 : f32
    }
    memref.dealloc %33 : memref<4x2xf32>
    %35 = memref.alloc() : memref<4xf32>
    linalg.fill(%cst, %35) : f32, memref<4xf32> 
    %36 = memref.alloc() : memref<4xf32>
    memref.copy %35, %36 : memref<4xf32> to memref<4xf32>
    memref.dealloc %35 : memref<4xf32>
    linalg.generic {indexing_maps = [#map4, #map6], iterator_types = ["parallel", "reduction"]} ins(%34 : memref<4x2xf32>) outs(%36 : memref<4xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      %41 = arith.addf %arg2, %arg3 : f32
      linalg.yield %41 : f32
    }
    %37 = memref.expand_shape %36 [[0, 1]] : memref<4xf32> into memref<4x1xf32>
    %38 = memref.alloc() : memref<4x1xf32>
    linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%37 : memref<4x1xf32>) outs(%38 : memref<4x1xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      %41 = arith.divf %cst_1, %arg2 : f32
      linalg.yield %41 : f32
    }
    memref.dealloc %36 : memref<4xf32>
    %39 = memref.collapse_shape %38 [[0, 1]] : memref<4x1xf32> into memref<4xf32>
    %40 = memref.alloc() : memref<4x2xf32>
    linalg.generic {indexing_maps = [#map4, #map6, #map4], iterator_types = ["parallel", "parallel"]} ins(%34, %39 : memref<4x2xf32>, memref<4xf32>) outs(%40 : memref<4x2xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %41 = arith.mulf %arg2, %arg3 : f32
      linalg.yield %41 : f32
    }
    memref.dealloc %38 : memref<4x1xf32>
    memref.dealloc %34 : memref<4x2xf32>
    memref.copy %40, %arg1 : memref<4x2xf32> to memref<4x2xf32>
    return
  }
}

