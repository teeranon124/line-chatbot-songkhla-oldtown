# -*- coding: utf-8 -*-
"""
Script to compile and export verified Google Knowledge Panel facts
for Songkhla Old Town attractions, food spots, and hotels.
Outputs:
1. songkhla_places_facts.json (Structured data for Knowledge Graph & LINE Bot)
2. Songkhla_Old_Town_Factsheet.md (Enriched text for Dense RAG chunking)
"""

import json
import os

PLACES_DATA = [
    {
        "id": "aitim_oang",
        "name": "ร้านไอติมโอ่ง",
        "name_en": "Aitim Oang (Ong Ice Cream)",
        "category": "ของหวาน / เครื่องดื่ม",
        "street": "ถนนนางงาม",
        "address": "ถนนนางงาม ซอย 2 ต.บ่อยาง อ.เมืองสงขลา จ.สงขลา 90000",
        "landmark_clue": "ตั้งอยู่ตรงข้ามศาลเจ้าพ่อหลักเมืองสงขลา",
        "open_hours": "10:00 - 18:30 น.",
        "open_days": "เปิดทุกวัน (จันทร์ - อาทิตย์)",
        "price_range": "20 - 30 บาท",
        "rating": 4.3,
        "review_count": 1250,
        "phone": "074-315594",
        "signature_items": ["ไอติมไข่แข็ง", "ไอติมทรงเครื่องใส่ไข่", "ลูกชิ้นปลาทอด", "เกี๊ยวทอด"],
        "lat": 7.1966,
        "lon": 100.5908,
        "google_maps_url": "https://maps.google.com/?q=7.1966,100.5908",
        "anyflip_page": 24
    },
    {
        "id": "kiat_fang",
        "name": "ร้านเกียดฟั่ง (ข้าวสตูสงขลา)",
        "name_en": "Kiat Fang Stew & Salapao",
        "category": "อาหารคาว",
        "street": "ถนนนางงาม",
        "address": "94 ถนนนางงาม ต.บ่อยาง อ.เมืองสงขลา จ.สงขลา 90000",
        "landmark_clue": "สาขาแรกดั้งเดิมรุ่นที่ 3 ริมถนนนางงาม (เปิดมาตั้งแต่ พ.ศ. 2480)",
        "open_hours": "07:00 - 14:00 น. (หรือจนกว่าของจะหมด)",
        "open_days": "เปิดทุกวัน (จันทร์ - อาทิตย์)",
        "price_range": "60 - 150 บาท (ซาลาเปา 20 - 35 บาท)",
        "rating": 4.2,
        "review_count": 1840,
        "phone": "074-311998",
        "signature_items": ["ข้าวสตูหมูเครื่องใน", "หมูกรอบ", "ข้าวหมูแดง", "ซาลาเปาลูกใหญ่ไส้หมูสับ"],
        "lat": 7.1973,
        "lon": 100.5904,
        "google_maps_url": "https://maps.google.com/?q=7.1973,100.5904",
        "anyflip_page": 23
    },
    {
        "id": "tae_hiang_iu",
        "name": "ร้านแต้เฮี้ยงอิ้ว",
        "name_en": "Tae Hiang Iu Restaurant",
        "category": "อาหารคาว",
        "street": "ถนนนางงาม",
        "address": "85 ถนนนางงาม ต.บ่อยาง อ.เมืองสงขลา จ.สงขลา 90000",
        "landmark_clue": "ตั้งอยู่ตรงข้ามร้านบ้านขนมไทยสองแสน บนถนนนางงาม",
        "open_hours": "รอบเที่ยง 11:30 - 14:00 น. และ รอบเย็น 17:00 - 20:00 น.",
        "open_days": "เปิดทุกวัน (จันทร์ - อาทิตย์)",
        "price_range": "150 - 400 บาท",
        "rating": 4.4,
        "review_count": 920,
        "phone": "074-311505",
        "signature_items": ["ต้มยำแห้งปลากระพง", "เป็ดพะโล้", "กุ้งแม่น้ำทอดกระเทียม", "หมูสับต้มบ๊วย", "เต้าหู้ราดหน้าปู"],
        "lat": 7.1969,
        "lon": 100.5906,
        "google_maps_url": "https://maps.google.com/?q=7.1969,100.5906",
        "anyflip_page": 25
    },
    {
        "id": "jae_ni",
        "name": "ร้านเจ๊นิ ข้าวต้มปลา (สาขาโรงสีแดง)",
        "name_en": "Jae Ni Fish Porridge (Hub Ho Hin Branch)",
        "category": "อาหารคาว",
        "street": "ถนนนครนอก",
        "address": "ถนนนครนอก ต.บ่อยาง อ.เมืองสงขลา จ.สงขลา 90000",
        "landmark_clue": "สาขาโรงสีแดง ตั้งอยู่เยื้องกับโรงสีแดงหับโห้หิ้น ย่านเมืองเก่า",
        "open_hours": "09:00 - 17:00 น.",
        "open_days": "เปิดทุกวัน (จันทร์ - อาทิตย์)",
        "price_range": "60 - 120 บาท",
        "rating": 4.3,
        "review_count": 510,
        "phone": "081-8979307",
        "signature_items": ["ข้าวต้มปลากะพงสด", "หมี่ซั่วแห้งรวมพล", "ข้าวต้มกระดูกหมู", "ลูกชิ้นปลาลวก"],
        "lat": 7.1983,
        "lon": 100.5886,
        "google_maps_url": "https://maps.google.com/?q=7.1983,100.5886",
        "anyflip_page": 28
    },
    {
        "id": "khanom_thai_song_saen",
        "name": "บ้านขนมไทยสองแสน",
        "name_en": "Baan Khanom Thai Song-Saen",
        "category": "ของหวาน / ของฝาก",
        "street": "ถนนนางงาม",
        "address": "ถนนนางงาม ต.บ่อยาง อ.เมืองสงขลา จ.สงขลา 90000",
        "landmark_clue": "อยู่ตรงข้ามร้านแต้เฮี้ยงอิ้ว ใกล้ศาลเจ้าพ่อหลักเมือง",
        "open_hours": "08:00 - 19:30 น. (หยุดทุกวันพุธ)",
        "open_days": "วันพฤหัสบดี - วันอังคาร (ปิดวันพุธ)",
        "price_range": "30 - 100 บาท",
        "rating": 4.5,
        "review_count": 680,
        "phone": "074-321588 / 061-3695351",
        "signature_items": ["ขนมทองเอก", "ขนมสัมปันนี", "ขนมขี้มอด", "สาเกเชื่อม", "ข้าวฟ่างเปียกกะทิสด"],
        "lat": 7.1968,
        "lon": 100.5907,
        "google_maps_url": "https://maps.google.com/?q=7.1968,100.5907",
        "anyflip_page": 29
    },
    {
        "id": "songkhla_station",
        "name": "สงขลาสเตชั่น (Songkhla Station Cafe & Art)",
        "name_en": "Songkhla Station Cafe & Art Space",
        "category": "ของหวาน / คาเฟ่และศิลปะ",
        "street": "ถนนนครนอก",
        "address": "ถนนนครนอก ต.บ่อยาง อ.เมืองสงขลา จ.สงขลา 90000",
        "landmark_clue": "อาคารตึกไม้โบราณ 2 ชั้น เยื้องโรงสีแดงหับโห้หิ้น ชั้นสองมีหอศิลป์",
        "open_hours": "จันทร์-ศุกร์ 09:30 - 18:00 น. / เสาร์-อาทิตย์ 08:00 - 19:00 น.",
        "open_days": "เปิดทุกวัน (จันทร์ - อาทิตย์)",
        "price_range": "50 - 120 บาท",
        "rating": 4.4,
        "review_count": 430,
        "phone": "081-8984088 / 093-7711181",
        "signature_items": ["กาแฟสดสเปเชียลตี้", "ชาไทยโบราณ", "วาฟเฟิลโฮมเมด", "นิทรรศการภาพถ่ายเมืองเก่า"],
        "lat": 7.1981,
        "lon": 100.5888,
        "google_maps_url": "https://maps.google.com/?q=7.1981,100.5888",
        "anyflip_page": 27
    },
    {
        "id": "hub_ho_hin",
        "name": "โรงสีแดง หับโห้หิ้น",
        "name_en": "Hub Ho Hin Red Rice Mill",
        "category": "สถานที่ท่องเที่ยวทางประวัติศาสตร์",
        "street": "ถนนนครนอก",
        "address": "13 ถนนนครนอก ต.บ่อยาง อ.เมืองสงขลา จ.สงขลา 90000",
        "landmark_clue": "อาคารไม้สีแดงสดริมทะเลสาบสงขลา ท่าเรือโบราณ",
        "open_hours": "08:00 - 18:00 น.",
        "open_days": "เปิดทุกวัน (จันทร์ - อาทิตย์)",
        "price_range": "เข้าชมฟรี (ไม่มีค่าใช้จ่าย)",
        "rating": 4.4,
        "review_count": 2150,
        "phone": "074-311015",
        "signature_items": ["ถ่ายภาพอาคารไม้สีแดงริมน้ำ", "นิทรรศการประวัติศาสตร์ท่าเรือสงขลา", "จุดชมวิวทะเลสาบสงขลา"],
        "lat": 7.1986,
        "lon": 100.5884,
        "google_maps_url": "https://maps.google.com/?q=7.1986,100.5884",
        "anyflip_page": 13
    },
    {
        "id": "baan_nakorn_in",
        "name": "บ้านนครใน",
        "name_en": "Baan Nakorn-In Heritage Museum",
        "category": "สถานที่ท่องเที่ยวทางประวัติศาสตร์",
        "street": "ถนนนครใน",
        "address": "117-122 ถนนนครใน ต.บ่อยาง อ.เมืองสงขลา จ.สงขลา 90000",
        "landmark_clue": "อาคารโบราณจีนและชิโนยูโรเปียน เชื่อมทะลุระหว่างถนนนครในกับถนนนครนอก",
        "open_hours": "08:00 - 18:00 น. (เสาร์-อาทิตย์ ปิด 19:00 น.)",
        "open_days": "เปิดทุกวัน (จันทร์ - อาทิตย์)",
        "price_range": "เข้าชมฟรี (ไม่มีค่าใช้จ่าย)",
        "rating": 4.5,
        "review_count": 1320,
        "phone": "089-8766388",
        "signature_items": ["พิพิธภัณฑ์ของสะสมโบราณ", "เตียงโบราณราชวงศ์ชิง", "สถาปัตยกรรมชิโน-ยูโรเปียน"],
        "lat": 7.1978,
        "lon": 100.5895,
        "google_maps_url": "https://maps.google.com/?q=7.1978,100.5895",
        "anyflip_page": 11
    },
    {
        "id": "baan_chinese_300yr",
        "name": "บ้านจีน 300 ปี",
        "name_en": "300-Year-Old Chinese House",
        "category": "สถานที่ท่องเที่ยวทางประวัติศาสตร์",
        "street": "ถนนนครนอก",
        "address": "ถนนนครนอก ต.บ่อยาง อ.เมืองสงขลา จ.สงขลา 90000",
        "landmark_clue": "บ้านไม้เก่าสไตล์จีนฮกเกี้ยนโบราณ ใกล้กับโรงสีแดง",
        "open_hours": "09:00 - 17:00 น.",
        "open_days": "เปิดทุกวัน (จันทร์ - อาทิตย์)",
        "price_range": "เข้าชมฟรี",
        "rating": 4.2,
        "review_count": 340,
        "phone": "-",
        "signature_items": ["สถาปัตยกรรมเรือนแถวไม้จีนดั้งเดิม", "ร่องรอยการค้าโบราณริมฝั่งทะเลสาบ"],
        "lat": 7.1980,
        "lon": 100.5887,
        "google_maps_url": "https://maps.google.com/?q=7.1980,100.5887",
        "anyflip_page": 14
    },
    {
        "id": "baan_ww2",
        "name": "บ้านสงครามโลก (ตึกสงครามโลก)",
        "name_en": "World War II Ruin House",
        "category": "สถานที่ท่องเที่ยวทางประวัติศาสตร์",
        "street": "ถนนนครใน",
        "address": "ถนนนครนอกตัดถนนยะลา ต.บ่อยาง อ.เมืองสงขลา จ.สงขลา 90000",
        "landmark_clue": "อาคารซากปรักหักพังประวัติศาสตร์จากระเบิดสงครามโลกครั้งที่ 2",
        "open_hours": "ชมภายนอกได้ตลอดเวลา / เปิดภายในเฉพาะช่วงจัดนิทรรศการศิลปะ",
        "open_days": "เปิดทุกวัน (ชมสถาปัตยกรรมภายนอก)",
        "price_range": "เข้าชมฟรี",
        "rating": 4.1,
        "review_count": 210,
        "phone": "-",
        "signature_items": ["โครงสร้างตึกประวัติศาสตร์ถูกทิ้งระเบิด พ.ศ. 2484", "จุดถ่ายภาพวินเทจ"],
        "lat": 7.1988,
        "lon": 100.5891,
        "google_maps_url": "https://maps.google.com/?q=7.1988,100.5891",
        "anyflip_page": 15
    },
    {
        "id": "city_pillar_shrine",
        "name": "ศาลเจ้าพ่อหลักเมืองสงขลา",
        "name_en": "Songkhla City Pillar Shrine",
        "category": "สถานที่ท่องเที่ยวทางวัฒนธรรม / สิ่งศักดิ์สิทธิ์",
        "street": "ถนนนางงาม",
        "address": "ถนนนางงาม ต.บ่อยาง อ.เมืองสงขลา จ.สงขลา 90000",
        "landmark_clue": "ศูนย์กลางความศรัทธาใจกลางถนนนางงาม ตรงข้ามร้านไอติมโอ่ง",
        "open_hours": "08:00 - 17:00 น.",
        "open_days": "เปิดทุกวัน (จันทร์ - อาทิตย์)",
        "price_range": "เข้าชมฟรี / ไหว้พระทำบุญตามศรัทธา",
        "rating": 4.6,
        "review_count": 2890,
        "phone": "074-311015",
        "signature_items": ["เสาหลักเมืองไม้ชัยพฤกษ์", "สถาปัตยกรรมศาลเจ้าจีนโบราณ", "สักการะเทพเจ้ากวนอูและเสี่ยงเซียมซี"],
        "lat": 7.1965,
        "lon": 100.5910,
        "google_maps_url": "https://maps.google.com/?q=7.1965,100.5910",
        "anyflip_page": 17
    },
    {
        "id": "songkhla_street_art",
        "name": "สงขลา สตรีทอาร์ท (Street Art)",
        "name_en": "Songkhla Old Town Street Art",
        "category": "จุดถ่ายภาพ / ศิลปะกลางแจ้ง",
        "street": "กระจายตามแนวถนนนางงาม ถนนนครนอก ถนนนครใน",
        "address": "ย่านเมืองเก่าสงขลา ต.บ่อยาง อ.เมืองสงขลา จ.สงขลา 90000",
        "landmark_clue": "ภาพวาดฝาผนังตึกเก่า เช่น ภาพร้านน้ำชา ฟอกซีย่า รถลาก เรือประมง",
        "open_hours": "ชมได้ตลอด 24 ชั่วโมง (แนะนำช่วง 08:00 - 17:30 น.)",
        "open_days": "เปิดทุกวัน",
        "price_range": "เข้าชมฟรี",
        "rating": 4.5,
        "review_count": 3100,
        "phone": "-",
        "signature_items": ["ภาพสตรีทอาร์ตร้านน้ำชาฮับเซ่ง", "ภาพเด็กเล่นริมกำแพง", "ภาพวิถีชีวิตชาวประมง"],
        "lat": 7.1972,
        "lon": 100.5902,
        "google_maps_url": "https://maps.google.com/?q=7.1972,100.5902",
        "anyflip_page": 18
    },
    {
        "id": "songkhla_art_center",
        "name": "หอศิลป์สงขลา (Songkhla Art Center)",
        "name_en": "Songkhla Art Center",
        "category": "พิพิธภัณฑ์และศิลปะ",
        "street": "ถนนกำแพงเพชร (ใกล้ถนนนางงาม)",
        "address": "14 ถนนกำแพงเพชร ต.บ่อยาง อ.เมืองสงขลา จ.สงขลา 90000",
        "landmark_clue": "อาคารโรงสีเก่าดัดแปลงเป็นหอศิลป์ ใกล้แนวกำแพงเมืองเก่า",
        "open_hours": "10:30 - 17:30 น. (ปิดทุกวันจันทร์)",
        "open_days": "วันอังคาร - วันอาทิตย์ (ปิดวันจันทร์)",
        "price_range": "เข้าชมฟรี (ไม่มีค่าใช้จ่าย)",
        "rating": 4.4,
        "review_count": 390,
        "phone": "081-5988899",
        "signature_items": ["นิทรรศการหมุนเวียนศิลปินท้องถิ่นและนานาชาติ", "สถาปัตยกรรมโครงสร้างเหล็กโรงสีเก่า"],
        "lat": 7.1958,
        "lon": 100.5915,
        "google_maps_url": "https://maps.google.com/?q=7.1958,100.5915",
        "anyflip_page": 10
    },
    {
        "id": "singora_tram",
        "name": "รถรางชมเมืองสงขลา (Singora Tram)",
        "name_en": "Singora Tram Tour",
        "category": "การเดินทางและกิจกรรมท่องเที่ยว",
        "street": "จุดเริ่มต้น: หน้าพิพิธภัณฑ์พธำมะรงค์ (ถ.จะนะ)",
        "address": "หน้าพิพิธภัณฑ์พธำมะรงค์ ถนนจะนะ ต.บ่อยาง อ.เมืองสงขลา จ.สงขลา 90000",
        "landmark_clue": "รถรางนำเที่ยวบริการฟรีโดยเทศบาลนครสงขลา มีวิทยากรบรรยายประวัติศาสตร์",
        "open_hours": "มี 6 รอบต่อวัน: 09:00, 10:00, 11:00, 13:00, 14:00, 15:00 น.",
        "open_days": "ให้บริการทุกวัน (รวมวันหยุดนักขัตฤกษ์)",
        "price_range": "นั่งฟรี ไม่มีค่าใช้จ่าย",
        "rating": 4.6,
        "review_count": 870,
        "phone": "074-311015 (เทศบาลนครสงขลา)",
        "signature_items": ["นั่งชม 3 ถนนเมืองเก่า", "แวะถ่ายรูปแหลมสมิหลา นางเงือกทอง", "ชมประติมากรรมพญานาคพ่นน้ำ"],
        "lat": 7.1995,
        "lon": 100.5930,
        "google_maps_url": "https://maps.google.com/?q=7.1995,100.5930",
        "anyflip_page": 20
    },
    {
        "id": "khao_tang_kuan",
        "name": "เขาตังกวน (ลิฟต์กระเช้าไฟฟ้าเขาตังกวน)",
        "name_en": "Khao Tang Kuan & Cable Lift",
        "category": "จุดชมวิว / สถานที่ท่องเที่ยวธรรมชาติและประวัติศาสตร์",
        "street": "ถนนสุขุม เชิงเขาตังกวน",
        "address": "ถนนสุขุม ต.บ่อยาง อ.เมืองสงขลา จ.สงขลา 90000",
        "landmark_clue": "ยอดเขาใจกลางเมืองสงขลา มีลิฟต์โดยสารขึ้นสู่ยอดเขาและพระเจดีย์หลวง",
        "open_hours": "08:30 - 18:30 น.",
        "open_days": "เปิดทุกวัน (จันทร์ - อาทิตย์)",
        "price_range": "ค่าลิฟต์ขึ้น-ลง ผู้ใหญ่ 30 บาท / เด็ก 20 บาท",
        "rating": 4.5,
        "review_count": 3450,
        "phone": "074-316330",
        "signature_items": ["วิว 360 องศาเห็นทั้งทะเลอ่าวไทยและทะเลสาบสงขลา", "พระเจดีย์หลวงคู่บ้านคู่เมือง", "ประภาคารโบราณสมัย ร.5"],
        "lat": 7.2104,
        "lon": 100.5898,
        "google_maps_url": "https://maps.google.com/?q=7.2104,100.5898",
        "anyflip_page": 21
    },
    {
        "id": "hotel_songkhla_taeraek",
        "name": "โรงแรมสงขลาแต่แรก (Songkhla TaeRaek Antique Hotel)",
        "name_en": "Songkhla TaeRaek Antique Hotel",
        "category": "โรงแรม / ที่พัก",
        "street": "ถนนเพชรคีรี (ย่านเมืองเก่า)",
        "address": "40 ถนนเพชรคีรี ต.บ่อยาง อ.เมืองสงขลา จ.สงขลา 90000",
        "landmark_clue": "โรงแรมสไตล์แอนทีคคลาสสิก อยู่ตรงหัวมุมถนนเพชรคีรี เดินถึงถนนนางงามได้ง่าย",
        "open_hours": "เปิด 24 ชั่วโมง (Check-in 14:00 น. / Check-out 12:00 น.)",
        "open_days": "เปิดทุกวัน",
        "price_range": "800 - 1,800 บาท/คืน",
        "rating": 4.4,
        "review_count": 520,
        "phone": "074-322227",
        "signature_items": ["ห้องพักตกแต่งย้อนยุควิคตอเรียนผสมชิโนโปรตุกีส", "ระเบียงชมวิวเมืองเก่า", "บริการจักรยานปั่นเที่ยวฟรี"],
        "lat": 7.1952,
        "lon": 100.5901,
        "google_maps_url": "https://maps.google.com/?q=7.1952,100.5901",
        "anyflip_page": 31
    },
    {
        "id": "hotel_club_tree",
        "name": "โรงแรมคลับทรี (Club Tree Hotel)",
        "name_en": "Club Tree Hotel",
        "category": "โรงแรม / ที่พัก",
        "street": "ถนนทะเลหลวง",
        "address": "165/8 ถนนทะเลหลวง ต.บ่อยาง อ.เมืองสงขลา จ.สงขลา 90000",
        "landmark_clue": "โรงแรมสไตล์บูทีกโมเดิร์นโคโลเนียล ใกล้หาดชลาทัศน์และเมืองเก่า",
        "open_hours": "เปิด 24 ชั่วโมง (Check-in 14:00 น. / Check-out 12:00 น.)",
        "open_days": "เปิดทุกวัน",
        "price_range": "1,000 - 1,300 บาท/คืน",
        "rating": 4.3,
        "review_count": 780,
        "phone": "098-1515515",
        "signature_items": ["ห้องพักโมเดิร์นสะอาด สงบ", "ที่จอดรถสะดวกสบาย", "ใกล้หาดสมิหลา"],
        "lat": 7.1912,
        "lon": 100.5982,
        "google_maps_url": "https://maps.google.com/?q=7.1912,100.5982",
        "anyflip_page": 32
    },
    {
        "id": "hotel_montana",
        "name": "โรงแรมมอนทาน่า สงขลา (Montana Hotel)",
        "name_en": "Montana Hotel Songkhla",
        "category": "โรงแรม / ที่พัก",
        "street": "ถนนสะเดา",
        "address": "24/3 ถนนสะเดา ต.บ่อยาง อ.เมืองสงขลา จ.สงขลา 90000",
        "landmark_clue": "โรงแรมทำเลสะดวก ใกล้สนามกีฬาติณสูลานนท์และหาดชลาทัศน์",
        "open_hours": "เปิด 24 ชั่วโมง (Check-in 14:00 น. / Check-out 12:00 น.)",
        "open_days": "เปิดทุกวัน",
        "price_range": "1,500 - 2,500 บาท/คืน (รวมอาหารเช้า)",
        "rating": 4.2,
        "review_count": 640,
        "phone": "074-300614 / 063-0819747",
        "signature_items": ["ห้องพักขนาดใหญ่พร้อมระเบียง", "บุฟเฟต์อาหารเช้า", "สระว่ายน้ำและห้องฟิตเนส"],
        "lat": 7.1925,
        "lon": 100.6015,
        "google_maps_url": "https://maps.google.com/?q=7.1925,100.6015",
        "anyflip_page": 32
    }
]


def export_data():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    os.makedirs(base_dir, exist_ok=True)
    
    # 1. Export JSON
    json_path = os.path.join(base_dir, "songkhla_places_facts.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(PLACES_DATA, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved {len(PLACES_DATA)} places to {json_path}")
    
    # 2. Export Markdown Factsheet (for Vector RAG Chunking)
    md_path = os.path.join(base_dir, "Songkhla_Old_Town_Factsheet.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# ข้อมูลข้อเท็จจริงสถานที่ท่องเที่ยว ย่านเมืองเก่าสงขลา (Songkhla Old Town Verified Factsheet)\n\n")
        f.write("> เอกสารรวบรวมข้อมูลข้อเท็จจริง (เวลาเปิด-ปิด, พิกัดถนน, ราคา, เมนูเด่น, เบอร์โทร) สำหรับระบบ Hybrid GraphRAG\n\n")
        
        for p in PLACES_DATA:
            f.write(f"## {p['name']} ({p['name_en']})\n")
            f.write(f"- **ประเภท**: {p['category']}\n")
            f.write(f"- **ถนน / ทำเล**: {p['street']}\n")
            f.write(f"- **ที่อยู่**: {p['address']}\n")
            f.write(f"- **จุดสังเกต**: {p['landmark_clue']}\n")
            f.write(f"- **เวลาเปิด-ปิด**: {p['open_hours']}\n")
            f.write(f"- **วันเปิดทำการ**: {p['open_days']}\n")
            f.write(f"- **ช่วงราคา / ค่าบริการ**: {p['price_range']}\n")
            f.write(f"- **คะแนนรีวิว**: {p['rating']} ดาว ({p['review_count']:,} รีวิว)\n")
            f.write(f"- **เบอร์โทรศัพท์**: {p['phone']}\n")
            f.write(f"- **เมนูเด่น / จุดเด่น**: {', '.join(p['signature_items'])}\n")
            f.write(f"- **พิกัด GPS**: ละติจูด {p['lat']}, ลองจิจูด {p['lon']}\n")
            f.write(f"- **แผนที่ Google Maps**: {p['google_maps_url']}\n")
            f.write(f"- **อ้างอิงคู่มือ AnyFlip**: หน้า {p['anyflip_page']}\n\n")
            f.write("---\n\n")
            
    print(f"[OK] Saved Markdown Factsheet to {md_path}")


if __name__ == "__main__":
    export_data()
