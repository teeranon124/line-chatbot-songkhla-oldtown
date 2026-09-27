# -*- coding: utf-8 -*-
"""
LINE Flex Message Templates Module for Songkhla Old Town Assistant.
Provides high-fidelity, interactive Flex Bubble & Carousel Cards:
1. Welcome Hub & Main Menu Card
2. Food & Dessert Interactive Carousel (with Maps & Call buttons)
3. Historical Attractions Carousel
4. 2-Day Itinerary Carousel
5. Hotels & Accommodations Carousel
6. Emergency Services Card (with Direct Dial buttons)
7. AI Smart Answer Card (with Model Badge, Latency, and Citations)
"""
import re
from typing import List, Dict, Any
from linebot.models import (
    FlexSendMessage,
    QuickReply,
    QuickReplyButton,
    MessageAction
)

# Cloudflare / Public Images for authentic Songkhla Old Town
PLACE_IMAGES = {
    "aitim_oang": "https://online.anyflip.com/clfen/iaoe/files/large/24.webp",
    "kiat_fang": "https://online.anyflip.com/clfen/iaoe/files/large/23.webp",
    "tae_hiang_iu": "https://online.anyflip.com/clfen/iaoe/files/large/25.webp",
    "jae_ni": "https://online.anyflip.com/clfen/iaoe/files/large/28.webp",
    "khanom_thai_song_saen": "https://online.anyflip.com/clfen/iaoe/files/large/29.webp",
    "songkhla_station": "https://online.anyflip.com/clfen/iaoe/files/large/27.webp",
    "hub_ho_hin": "https://online.anyflip.com/clfen/iaoe/files/large/13.webp",
    "baan_nakorn_in": "https://online.anyflip.com/clfen/iaoe/files/large/11.webp",
    "baan_chinese_300yr": "https://online.anyflip.com/clfen/iaoe/files/large/15.webp",
    "baan_ww2": "https://online.anyflip.com/clfen/iaoe/files/large/16.webp",
    "city_pillar_shrine": "https://online.anyflip.com/clfen/iaoe/files/large/17.webp",
    "songkhla_street_art": "https://online.anyflip.com/clfen/iaoe/files/large/18.webp",
    "songkhla_art_center": "https://online.anyflip.com/clfen/iaoe/files/large/10.webp",
    "singora_tram": "https://online.anyflip.com/clfen/iaoe/files/large/20.webp",
    "khao_tang_kuan": "https://online.anyflip.com/clfen/iaoe/files/large/21.webp",
    "hotel_songkhla_taeraek": "https://online.anyflip.com/clfen/iaoe/files/large/31.webp",
    "hotel_club_tree": "https://online.anyflip.com/clfen/iaoe/files/large/32.webp",
    "hotel_montana": "https://online.anyflip.com/clfen/iaoe/files/large/32.webp",
    "default_banner": "https://online.anyflip.com/clfen/iaoe/files/large/1.webp"
}


class SongkhlaFlexTemplates:

    @staticmethod
    def clean_tel_uri(phone_raw: str, default: str = "tel:074311015") -> str:
        """Sanitizes phone strings into strictly valid RFC 3986 tel: URI with digits only."""
        if not phone_raw or str(phone_raw).strip() in ["-", ""]:
            return default
        first_part = re.split(r"[/,]", str(phone_raw))[0]
        digits = re.sub(r"\D", "", first_part)
        if len(digits) >= 8:
            return f"tel:{digits}"
        return default

    @staticmethod
    def clean_maps_url(url_raw: str, default: str = "https://maps.google.com/?q=Songkhla+Old+Town") -> str:
        """Ensures URL strictly conforms to HTTP/HTTPS scheme required by LINE."""
        if not url_raw or not isinstance(url_raw, str):
            return default
        url_clean = url_raw.strip()
        if url_clean.startswith("http://") or url_clean.startswith("https://"):
            return url_clean
        return default

    @staticmethod
    def get_quick_replies(custom_items=None) -> QuickReply:
        """Standard or Context-Aware Quick Reply chips below the chat box."""
        if custom_items:
            items = custom_items[:6]
        else:
            items = [
                ("🧭 หน้าแรกนำทาง", "หน้าแรก"),
                ("🗺️ แผน 2 วัน 1 คืน", "ขอแผนเที่ยว 2 วัน 1 คืน"),
                ("🍜 สำรวจร้านอาหาร", "สำรวจร้านอาหารทั้งหมด"),
                ("🏛️ สำรวจที่เที่ยว", "สำรวจที่เที่ยวทั้งหมด"),
                ("🏨 สำรวจที่พัก", "สำรวจที่พักทั้งหมด"),
                ("🚨 สายด่วนฉุกเฉิน", "ขอเบอร์ฉุกเฉิน")
            ]
        buttons = [
            QuickReplyButton(action=MessageAction(label=label, text=text))
            for label, text in items
        ]
        return QuickReply(items=buttons)

    @staticmethod
    def build_welcome_hub() -> FlexSendMessage:
        """Main Welcome Hub card."""
        bubble = {
            "type": "bubble",
            "hero": {
                "type": "image",
                "url": PLACE_IMAGES["default_banner"],
                "size": "full",
                "aspectRatio": "20:13",
                "aspectMode": "cover"
            },
            "body": {
                "type": "box",
                "layout": "vertical",
                "contents": [
                    {
                        "type": "text",
                        "text": "🏮 น้องสิงขร AI Guide",
                        "weight": "bold",
                        "size": "xl",
                        "color": "#1a1a2e"
                    },
                    {
                        "type": "text",
                        "text": "ศูนย์ข้อมูลและนำเที่ยวย่านเมืองเก่าสงขลา",
                        "size": "sm",
                        "color": "#0984e3",
                        "margin": "xs"
                    },
                    {
                        "type": "separator",
                        "margin": "md"
                    },
                    {
                        "type": "text",
                        "text": "ยินดีต้อนรับสู่ 3 ถนนสายวัฒนธรรม (นครนอก, นครใน, นางงาม) คุณสามารถพิมพ์ถามคำถามเป็นภาษาธรรมชาติ หรือเลือกหัวข้อสำรวจด้านล่างได้เลยครับ:",
                        "size": "sm",
                        "color": "#555555",
                        "wrap": True,
                        "margin": "md"
                    },
                    {
                        "type": "box",
                        "layout": "vertical",
                        "margin": "lg",
                        "spacing": "sm",
                        "contents": [
                            {
                                "type": "button",
                                "style": "primary",
                                "color": "#e84118",
                                "height": "sm",
                                "action": {
                                    "type": "message",
                                    "label": "🗺️ แผนเที่ยว 2 วัน 1 คืน",
                                    "text": "ขอแผนเที่ยว 2 วัน 1 คืน"
                                }
                            },
                            {
                                "type": "button",
                                "style": "secondary",
                                "height": "sm",
                                "action": {
                                    "type": "message",
                                    "label": "🍜 สำรวจร้านอาหาร & ของหวานทั้งหมด",
                                    "text": "สำรวจร้านอาหารทั้งหมด"
                                }
                            },
                            {
                                "type": "button",
                                "style": "secondary",
                                "height": "sm",
                                "action": {
                                    "type": "message",
                                    "label": "🏛️ สำรวจสถานที่ท่องเที่ยว & จุดเช็คอิน",
                                    "text": "สำรวจที่เที่ยวทั้งหมด"
                                }
                            },
                            {
                                "type": "button",
                                "style": "secondary",
                                "height": "sm",
                                "action": {
                                    "type": "message",
                                    "label": "🏨 สำรวจโรงแรม & ที่พักแนะนำ",
                                    "text": "สำรวจที่พักทั้งหมด"
                                }
                            }
                        ]
                    }
                ]
            }
        }
        return FlexSendMessage(
            alt_text="🧭 ศูนย์นำเที่ยวย่านเมืองเก่าสงขลา - น้องสิงขร AI Guide",
            contents=bubble,
            quick_reply=SongkhlaFlexTemplates.get_quick_replies()
        )

    @staticmethod
    def build_food_carousel(places: List[Dict[str, Any]]) -> FlexSendMessage:
        """Food and Dessert Carousel Cards."""
        bubbles = []
        for p in places:
            pid = p.get("id", "")
            img_url = PLACE_IMAGES.get(pid, PLACE_IMAGES["default_banner"])
            maps_url = SongkhlaFlexTemplates.clean_maps_url(p.get("google_maps_url"))
            tel_uri = SongkhlaFlexTemplates.clean_tel_uri(p.get("phone", ""), default="tel:074311015")
            
            dishes = p.get("signature_items", [])
            dishes_str = ", ".join(dishes[:3]) if dishes else "อาหารรสเลิศ"

            bubble = {
                "type": "bubble",
                "size": "kilo",
                "hero": {
                    "type": "image",
                    "url": img_url,
                    "size": "full",
                    "aspectRatio": "16:10",
                    "aspectMode": "cover"
                },
                "body": {
                    "type": "box",
                    "layout": "vertical",
                    "contents": [
                        {
                            "type": "text",
                            "text": p.get("name", ""),
                            "weight": "bold",
                            "size": "md",
                            "wrap": True,
                            "color": "#1a1a2e"
                        },
                        {
                            "type": "box",
                            "layout": "horizontal",
                            "contents": [
                                {
                                    "type": "text",
                                    "text": f"⭐ {p.get('rating', 4.3)} ({p.get('review_count', 0):,} รีวิว)",
                                    "size": "xs",
                                    "color": "#e67e22",
                                    "weight": "bold"
                                },
                                {
                                    "type": "text",
                                    "text": f"💰 {p.get('price_range', '')}",
                                    "size": "xs",
                                    "color": "#27ae60",
                                    "align": "end"
                                }
                            ],
                            "margin": "xs"
                        },
                        {
                            "type": "text",
                            "text": f"🕒 {p.get('open_hours', '')}",
                            "size": "xs",
                            "color": "#7f8c8d",
                            "margin": "xs"
                        },
                        {
                            "type": "text",
                            "text": f"📍 {p.get('street', '')} ({p.get('landmark_clue', '')})",
                            "size": "xs",
                            "color": "#34495e",
                            "wrap": True,
                            "margin": "xs"
                        },
                        {
                            "type": "separator",
                            "margin": "sm"
                        },
                        {
                            "type": "text",
                            "text": f"🍴 เมนูเด่น: {dishes_str}",
                            "size": "xs",
                            "color": "#d35400",
                            "wrap": True,
                            "margin": "sm"
                        }
                    ]
                },
                "footer": {
                    "type": "box",
                    "layout": "horizontal",
                    "spacing": "sm",
                    "contents": [
                        {
                            "type": "button",
                            "style": "primary",
                            "color": "#2980b9",
                            "height": "sm",
                            "action": {
                                "type": "uri",
                                "label": "📍 แผนที่",
                                "uri": maps_url
                            }
                        },
                        {
                            "type": "button",
                            "style": "secondary",
                            "height": "sm",
                            "action": {
                                "type": "uri",
                                "label": "📞 โทร",
                                "uri": tel_uri
                            }
                        }
                    ]
                }
            }
            bubbles.append(bubble)

        carousel = {"type": "carousel", "contents": bubbles[:10]}
        return FlexSendMessage(
            alt_text="🍜 รวมร้านอาหารและของหวานในตำนาน ย่านเมืองเก่าสงขลา",
            contents=carousel,
            quick_reply=SongkhlaFlexTemplates.get_quick_replies()
        )

    @staticmethod
    def build_attractions_carousel(places: List[Dict[str, Any]]) -> FlexSendMessage:
        """Historical Attractions Carousel."""
        bubbles = []
        for p in places:
            pid = p.get("id", "")
            img_url = PLACE_IMAGES.get(pid, PLACE_IMAGES["default_banner"])
            maps_url = p.get("google_maps_url", "https://maps.google.com/?q=Songkhla")

            bubble = {
                "type": "bubble",
                "size": "kilo",
                "hero": {
                    "type": "image",
                    "url": img_url,
                    "size": "full",
                    "aspectRatio": "16:10",
                    "aspectMode": "cover"
                },
                "body": {
                    "type": "box",
                    "layout": "vertical",
                    "contents": [
                        {
                            "type": "text",
                            "text": p.get("name", ""),
                            "weight": "bold",
                            "size": "md",
                            "wrap": True
                        },
                        {
                            "type": "text",
                            "text": f"🏛️ {p.get('street', '')}",
                            "size": "xs",
                            "color": "#2980b9",
                            "weight": "bold",
                            "margin": "xs"
                        },
                        {
                            "type": "text",
                            "text": f"🕒 เวลา: {p.get('open_hours', '')}",
                            "size": "xs",
                            "color": "#7f8c8d",
                            "margin": "xs"
                        },
                        {
                            "type": "text",
                            "text": f"🎟️ ค่าเข้าชม: {p.get('price_range', '')}",
                            "size": "xs",
                            "color": "#27ae60",
                            "margin": "xs"
                        },
                        {
                            "type": "separator",
                            "margin": "sm"
                        },
                        {
                            "type": "text",
                            "text": f"✨ ไฮไลต์: {p.get('landmark_clue', '')}",
                            "size": "xs",
                            "color": "#555555",
                            "wrap": True,
                            "margin": "sm"
                        }
                    ]
                },
                "footer": {
                    "type": "box",
                    "layout": "horizontal",
                    "spacing": "sm",
                    "contents": [
                        {
                            "type": "button",
                            "style": "primary",
                            "color": "#8e44ad",
                            "height": "sm",
                            "action": {
                                "type": "uri",
                                "label": "📍 แผนที่นำทาง",
                                "uri": maps_url
                            }
                        }
                    ]
                }
            }
            bubbles.append(bubble)

        carousel = {"type": "carousel", "contents": bubbles[:10]}
        return FlexSendMessage(
            alt_text="🏛️ สถานที่ท่องเที่ยวและจุดเช็คอิน ย่านเมืองเก่าสงขลา",
            contents=carousel,
            quick_reply=SongkhlaFlexTemplates.get_quick_replies()
        )

    @staticmethod
    def build_itinerary_carousel() -> FlexSendMessage:
        """2-Day 1-Night Itinerary Cards."""
        card_day1 = {
            "type": "bubble",
            "size": "mega",
            "hero": {
                "type": "image",
                "url": PLACE_IMAGES["default_banner"],
                "size": "full",
                "aspectRatio": "20:10",
                "aspectMode": "cover"
            },
            "body": {
                "type": "box",
                "layout": "vertical",
                "contents": [
                    {
                        "type": "text",
                        "text": "โปรแกรมวันที่ 1: เที่ยวรอบเมืองเก่า 3 ถนน",
                        "weight": "bold",
                        "size": "md",
                        "color": "#c0392b"
                    },
                    {
                        "type": "separator",
                        "margin": "sm"
                    },
                    {
                        "type": "box",
                        "layout": "vertical",
                        "margin": "md",
                        "spacing": "xs",
                        "contents": [
                            {"type": "text", "text": "• 09.00 น. หอศิลป์สงขลา (Songkhla Art Center)", "size": "xs", "color": "#333333"},
                            {"type": "text", "text": "• 10.00 น. บ้านนครใน (พิพิธภัณฑ์บ้านโบราณ)", "size": "xs", "color": "#333333"},
                            {"type": "text", "text": "• 11.00 น. โรงสีแดงหับโห้หิ้น (ถ่ายรูปริมน้ำ)", "size": "xs", "color": "#333333"},
                            {"type": "text", "text": "• 12.00 น. มื้อเที่ยง: ข้าวสตู ร้านเกียดฟั่ง", "size": "xs", "color": "#d35400", "weight": "bold"},
                            {"type": "text", "text": "• 13.00 น. บ้านจีน 300 ปี (สถาปัตยกรรมฮกเกี้ยน)", "size": "xs", "color": "#333333"},
                            {"type": "text", "text": "• 14.00 น. เช็คอิน โรงแรมสงขลาแต่แรก", "size": "xs", "color": "#2980b9"},
                            {"type": "text", "text": "• 15.00 น. บ้านสงครามโลก ครั้งที่ 2", "size": "xs", "color": "#333333"},
                            {"type": "text", "text": "• 16.00 น. สักการะศาลเจ้าพ่อหลักเมืองสงขลา", "size": "xs", "color": "#333333"},
                            {"type": "text", "text": "• 17.00 น. สตรีทอาร์ท & แวะชิม ร้านไอติมโอ่ง", "size": "xs", "color": "#d35400", "weight": "bold"},
                            {"type": "text", "text": "• 18.00 น. มื้อค่ำ: ต้มยำแห้ง ร้านแต้เฮี้ยงอิ้ว", "size": "xs", "color": "#d35400", "weight": "bold"}
                        ]
                    }
                ]
            }
        }

        card_day2 = {
            "type": "bubble",
            "size": "mega",
            "hero": {
                "type": "image",
                "url": PLACE_IMAGES["singora_tram"],
                "size": "full",
                "aspectRatio": "20:10",
                "aspectMode": "cover"
            },
            "body": {
                "type": "box",
                "layout": "vertical",
                "contents": [
                    {
                        "type": "text",
                        "text": "โปรแกรมวันที่ 2: รถราง สมิหลา เขาตังกวน",
                        "weight": "bold",
                        "size": "md",
                        "color": "#27ae60"
                    },
                    {
                        "type": "separator",
                        "margin": "sm"
                    },
                    {
                        "type": "box",
                        "layout": "vertical",
                        "margin": "md",
                        "spacing": "xs",
                        "contents": [
                            {"type": "text", "text": "• 07.00 น. รับประทานอาหารเช้าที่โรงแรม", "size": "xs", "color": "#333333"},
                            {"type": "text", "text": "• 08.00 น. นั่งรถรางชมเมืองสงขลา ฟรี ชมพญานาค & นางเงือกทอง", "size": "xs", "color": "#2980b9", "weight": "bold"},
                            {"type": "text", "text": "• 10.00 น. ขึ้นลิฟต์กระเช้าไฟฟ้า ยอดเขาตังกวน (วิว 360 องศา)", "size": "xs", "color": "#333333"},
                            {"type": "text", "text": "• 11.00 น. พักจิบกาแฟ คาเฟ่ Songkhla Station", "size": "xs", "color": "#d35400"},
                            {"type": "text", "text": "• 12.00 น. มื้อเที่ยง: ข้าวต้มปลากะพง ร้านเจ๊นิ (สาขาโรงสีแดง)", "size": "xs", "color": "#d35400", "weight": "bold"},
                            {"type": "text", "text": "• 13.00 น. ซื้อของฝากทองเอก สัมปันนี บ้านขนมไทยสองแสน", "size": "xs", "color": "#d35400", "weight": "bold"},
                            {"type": "text", "text": "• 14.00 น. เดินทางกลับสนามบินหาดใหญ่โดยสวัสดิภาพ", "size": "xs", "color": "#27ae60"}
                        ]
                    }
                ]
            }
        }

        carousel = {"type": "carousel", "contents": [card_day1, card_day2]}
        return FlexSendMessage(
            alt_text="🗺️ แผนการท่องเที่ยวสงขลา 2 วัน 1 คืน",
            contents=carousel,
            quick_reply=SongkhlaFlexTemplates.get_quick_replies()
        )

    @staticmethod
    def build_hotels_carousel(hotels: List[Dict[str, Any]]) -> FlexSendMessage:
        """Hotels Carousel Cards."""
        bubbles = []
        for h in hotels:
            pid = h.get("id", "")
            img_url = PLACE_IMAGES.get(pid, PLACE_IMAGES["default_banner"])
            maps_url = SongkhlaFlexTemplates.clean_maps_url(h.get("google_maps_url"))
            tel_uri = SongkhlaFlexTemplates.clean_tel_uri(h.get("phone", ""), default="tel:074322227")

            bubble = {
                "type": "bubble",
                "size": "kilo",
                "hero": {
                    "type": "image",
                    "url": img_url,
                    "size": "full",
                    "aspectRatio": "16:10",
                    "aspectMode": "cover"
                },
                "body": {
                    "type": "box",
                    "layout": "vertical",
                    "contents": [
                        {
                            "type": "text",
                            "text": h.get("name", ""),
                            "weight": "bold",
                            "size": "md",
                            "wrap": True
                        },
                        {
                            "type": "text",
                            "text": f"🏨 {h.get('street', '')}",
                            "size": "xs",
                            "color": "#2980b9",
                            "weight": "bold",
                            "margin": "xs"
                        },
                        {
                            "type": "text",
                            "text": f"💰 ราคา: {h.get('price_range', '')}",
                            "size": "xs",
                            "color": "#27ae60",
                            "margin": "xs"
                        },
                        {
                            "type": "text",
                            "text": f"⭐ รีวิว: {h.get('rating', 4.3)} ดาว",
                            "size": "xs",
                            "color": "#e67e22",
                            "margin": "xs"
                        },
                        {
                            "type": "separator",
                            "margin": "sm"
                        },
                        {
                            "type": "text",
                            "text": f"📍 ทำเล: {h.get('landmark_clue', '')}",
                            "size": "xs",
                            "color": "#555555",
                            "wrap": True,
                            "margin": "sm"
                        }
                    ]
                },
                "footer": {
                    "type": "box",
                    "layout": "horizontal",
                    "spacing": "sm",
                    "contents": [
                        {
                            "type": "button",
                            "style": "primary",
                            "color": "#2980b9",
                            "height": "sm",
                            "action": {
                                "type": "uri",
                                "label": "📍 แผนที่",
                                "uri": maps_url
                            }
                        },
                        {
                            "type": "button",
                            "style": "secondary",
                            "height": "sm",
                            "action": {
                                "type": "uri",
                                "label": "📞 จองห้อง",
                                "uri": tel_uri
                            }
                        }
                    ]
                }
            }
            bubbles.append(bubble)

        carousel = {"type": "carousel", "contents": bubbles[:5]}
        return FlexSendMessage(
            alt_text="🏨 โรงแรมและที่พักยอดนิยม ย่านเมืองเก่าสงขลา",
            contents=carousel,
            quick_reply=SongkhlaFlexTemplates.get_quick_replies()
        )

    @staticmethod
    def build_emergency_card() -> FlexSendMessage:
        """Emergency and Tourist Hotline Card."""
        bubble = {
            "type": "bubble",
            "body": {
                "type": "box",
                "layout": "vertical",
                "contents": [
                    {
                        "type": "text",
                        "text": "🚨 เบอร์โทรสำคัญและสายด่วนฉุกเฉิน",
                        "weight": "bold",
                        "size": "md",
                        "color": "#c0392b"
                    },
                    {
                        "type": "text",
                        "text": "จังหวัดสงขลา พร้อมโทรออกได้ทันที",
                        "size": "xs",
                        "color": "#7f8c8d",
                        "margin": "xs"
                    },
                    {
                        "type": "separator",
                        "margin": "md"
                    },
                    {
                        "type": "box",
                        "layout": "vertical",
                        "margin": "md",
                        "spacing": "sm",
                        "contents": [
                            {
                                "type": "button",
                                "style": "primary",
                                "color": "#27ae60",
                                "height": "sm",
                                "action": {"type": "uri", "label": "👮 ตำรวจท่องเที่ยว (1155)", "uri": "tel:1155"}
                            },
                            {
                                "type": "button",
                                "style": "primary",
                                "color": "#2980b9",
                                "height": "sm",
                                "action": {"type": "uri", "label": "🚓 สภ.เมืองสงขลา (191)", "uri": "tel:191"}
                            },
                            {
                                "type": "button",
                                "style": "primary",
                                "color": "#e67e22",
                                "height": "sm",
                                "action": {"type": "uri", "label": "🚑 กู้ภัยเทศบาลสงขลา (074-311015)", "uri": "tel:074311015"}
                            },
                            {
                                "type": "button",
                                "style": "secondary",
                                "height": "sm",
                                "action": {"type": "uri", "label": "🏥 โรงพยาบาลสงขลา (074-338100)", "uri": "tel:074338100"}
                            }
                        ]
                    }
                ]
            }
        }
        return FlexSendMessage(
            alt_text="🚨 เบอร์โทรสำคัญและสายด่วนฉุกเฉินสงขลา",
            contents=bubble,
            quick_reply=SongkhlaFlexTemplates.get_quick_replies()
        )

    @staticmethod
    def extract_matched_places(text: str, places: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extracts specific POI places mentioned in query or answer text."""
        if not places or not text:
            return []
        text_lower = text.lower()
        matched = []
        seen_ids = set()

        aliases = [
            ("ไอติมโอ่ง", "aitim_oang"),
            ("เกียดฟั่ง", "kiat_fang"),
            ("สตูสงขลา", "kiat_fang"),
            ("แต้เฮี้ยงอิ้ว", "tae_hiang_iu"),
            ("เจ๊นิ", "jae_ni"),
            ("สองแสน", "khanom_thai_song_saen"),
            ("สอง-แสน", "khanom_thai_song_saen"),
            ("สงขลาสเตชั่น", "songkhla_station"),
            ("หับโห้หิ้น", "hub_ho_hin"),
            ("โรงสีแดง", "hub_ho_hin"),
            ("บ้านนครใน", "baan_nakorn_in"),
            ("บ้านจีน", "baan_chinese_300yr"),
            ("บ้านสงครามโลก", "baan_ww2"),
            ("ศาลเจ้าพ่อหลักเมือง", "city_pillar_shrine"),
            ("สตรีทอาร์ท", "songkhla_street_art"),
            ("street art", "songkhla_street_art"),
            ("หอศิลป์", "songkhla_art_center"),
            ("รถราง", "singora_tram"),
            ("เขาตังกวน", "khao_tang_kuan"),
            ("สงขลาแต่แรก", "hotel_songkhla_taeraek"),
            ("คลับทรี", "hotel_club_tree"),
            ("club tree", "hotel_club_tree"),
            ("มอนทาน่า", "hotel_montana")
        ]

        # 1. Match by place name or ID
        for p in places:
            pid = p.get("id", "")
            clean_pname = p.get("name", "").split("(")[0].strip().lower()
            if (clean_pname in text_lower or pid in text_lower) and pid not in seen_ids:
                matched.append(p)
                seen_ids.add(pid)

        # 2. Match by aliases
        for al, pid in aliases:
            if al in text_lower and pid not in seen_ids:
                for p in places:
                    if p.get("id") == pid:
                        matched.append(p)
                        seen_ids.add(pid)
                        break

        # 3. If no specific place, check street
        if not matched:
            for st in ["ถนนนางงาม", "ถนนนครนอก", "ถนนนครใน"]:
                if st in text_lower:
                    for p in places:
                        if p.get("street") == st and p.get("id") not in seen_ids:
                            matched.append(p)
                            seen_ids.add(p.get("id"))
                    break

        return matched[:5]

    @staticmethod
    def build_matched_places_carousel(matched_places: List[Dict[str, Any]]) -> FlexSendMessage:
        """
        Builds a dedicated, interactive Carousel containing bubbles
        only for the specifically matched places (e.g. 1 to 5 places).
        """
        bubbles = []
        for p in matched_places[:5]:
            pid = p.get("id", "")
            img_url = PLACE_IMAGES.get(pid, PLACE_IMAGES["default_banner"])
            pname = p.get("name", "").split("(")[0].strip()
            maps_url = SongkhlaFlexTemplates.clean_maps_url(p.get("google_maps_url"))
            tel_uri = SongkhlaFlexTemplates.clean_tel_uri(p.get("phone", ""), default="tel:074311015")

            dishes = p.get("signature_items", [])
            dishes_str = ", ".join(dishes[:2]) if dishes else "ของดีเมืองสงขลา"

            bubble = {
                "type": "bubble",
                "size": "kilo",
                "hero": {
                    "type": "image",
                    "url": img_url,
                    "size": "full",
                    "aspectRatio": "16:10",
                    "aspectMode": "cover"
                },
                "body": {
                    "type": "box",
                    "layout": "vertical",
                    "contents": [
                        {
                            "type": "text",
                            "text": pname,
                            "weight": "bold",
                            "size": "md",
                            "color": "#1a1a2e",
                            "wrap": True
                        },
                        {
                            "type": "box",
                            "layout": "horizontal",
                            "contents": [
                                {
                                    "type": "text",
                                    "text": f"⭐ {p.get('rating', 4.3)}",
                                    "size": "xs",
                                    "color": "#e67e22",
                                    "weight": "bold"
                                },
                                {
                                    "type": "text",
                                    "text": f"💰 {p.get('price_range', '')}",
                                    "size": "xs",
                                    "color": "#27ae60",
                                    "align": "end"
                                }
                            ],
                            "margin": "xs"
                        },
                        {
                            "type": "text",
                            "text": f"🕒 {p.get('open_hours', '')}",
                            "size": "xs",
                            "color": "#7f8c8d",
                            "margin": "xs"
                        },
                        {
                            "type": "text",
                            "text": f"📍 {p.get('street', '')}",
                            "size": "xs",
                            "color": "#34495e",
                            "margin": "xs"
                        },
                        {
                            "type": "separator",
                            "margin": "sm"
                        },
                        {
                            "type": "text",
                            "text": f"🍴 {dishes_str}",
                            "size": "xs",
                            "color": "#d35400",
                            "wrap": True,
                            "margin": "sm"
                        }
                    ]
                },
                "footer": {
                    "type": "box",
                    "layout": "horizontal",
                    "spacing": "sm",
                    "contents": [
                        {
                            "type": "button",
                            "style": "primary",
                            "color": "#0984e3",
                            "height": "sm",
                            "action": {
                                "type": "uri",
                                "label": "📍 แผนที่",
                                "uri": maps_url
                            }
                        },
                        {
                            "type": "button",
                            "style": "secondary",
                            "height": "sm",
                            "action": {
                                "type": "uri",
                                "label": "📞 โทร",
                                "uri": tel_uri
                            }
                        }
                    ]
                }
            }
            bubbles.append(bubble)

        carousel = {"type": "carousel", "contents": bubbles}
        return FlexSendMessage(
            alt_text=f"📍 แนะนำ {len(bubbles)} สถานที่ที่เกี่ยวข้อง",
            contents=carousel,
            quick_reply=SongkhlaFlexTemplates.get_quick_replies()
        )

    @staticmethod
    def build_ai_answer_card(rag_result: Dict[str, Any], places: List[Dict[str, Any]] = None) -> FlexSendMessage:
        """
        AI Smart Answer Flex Card:
        Provides clear, beautifully formatted, compact AI response without markdown artifacts,
        featuring a modern Pill Badge, clean typography, Grounding citations,
        and Dynamic Context-Aware Action Buttons (Google Maps navigation, phone dialer, street explorer).
        """
        query = rag_result.get("query", "")
        raw_answer = rag_result.get("answer", "")
        provider = rag_result.get("provider", "ollama")
        model = rag_result.get("model", "")
        latency = rag_result.get("latency_seconds", 0.0)
        sources = rag_result.get("sources", [])

        # Clean markdown artifacts and boilerplate greetings
        clean_answer = raw_answer.strip()
        clean_answer = re.sub(r'\*\*(.*?)\*\*', r'\1', clean_answer)
        clean_answer = re.sub(r'\*(.*?)\*', r'\1', clean_answer)
        clean_answer = re.sub(r'#+\s*', '', clean_answer)
        clean_answer = re.sub(r'^\s*สวัสดี.*?(ค่ะ|ครับ)[!🏮\s]*\n*', '', clean_answer)
        clean_answer = clean_answer.strip()

        is_local = "ollama" in provider.lower()
        badge_text = f"💻 Local ({latency}s)" if is_local else f"⚡ Cloud ({latency}s)"
        badge_bg = "#e8f8f5" if is_local else "#fef9e7"
        badge_color = "#16a085" if is_local else "#d35400"

        source_texts = []
        for s in sources[:2]:
            short_s = s.replace("[Graph Subgraph] ", "").replace(" (AnyFlip หน้า ", " (หน้า ")
            source_texts.append({
                "type": "text",
                "text": f"• {short_s}",
                "size": "xxs",
                "color": "#7f8c8d",
                "wrap": True
            })

        # --- Context-Aware Entity Resolution ---
        combined_text = f"{query} {clean_answer}".lower()
        matched_place = None
        if places:
            for p in places:
                clean_pname = p.get("name", "").split("(")[0].strip()
                if clean_pname.lower() in combined_text or p.get("id", "").lower() in combined_text:
                    matched_place = p
                    break
                # Common aliases
                aliases = [
                    ("ไอติมโอ่ง", "aitim_oang"),
                    ("เกียดฟั่ง", "kiat_fang"),
                    ("สตูสงขลา", "kiat_fang"),
                    ("แต้เฮี้ยงอิ้ว", "tae_hiang_iu"),
                    ("เจ๊นิ", "jae_ni"),
                    ("สองแสน", "khanom_thai_song_saen"),
                    ("สอง-แสน", "khanom_thai_song_saen"),
                    ("สงขลาสเตชั่น", "songkhla_station"),
                    ("หับโห้หิ้น", "hub_ho_hin"),
                    ("โรงสีแดง", "hub_ho_hin"),
                    ("บ้านนครใน", "baan_nakorn_in"),
                    ("บ้านจีน", "baan_chinese_300yr"),
                    ("บ้านสงครามโลก", "baan_ww2"),
                    ("ศาลเจ้าพ่อหลักเมือง", "city_pillar_shrine"),
                    ("สตรีทอาร์ท", "songkhla_street_art"),
                    ("street art", "songkhla_street_art"),
                    ("หอศิลป์", "songkhla_art_center"),
                    ("รถราง", "singora_tram"),
                    ("เขาตังกวน", "khao_tang_kuan"),
                    ("สงขลาแต่แรก", "hotel_songkhla_taeraek"),
                    ("คลับทรี", "hotel_club_tree"),
                    ("club tree", "hotel_club_tree"),
                    ("มอนทาน่า", "hotel_montana")
                ]
                for al, pid in aliases:
                    if al in combined_text and p.get("id") == pid:
                        matched_place = p
                        break
                if matched_place:
                    break

        # Detect street context
        matched_street = None
        for st in ["ถนนนางงาม", "ถนนนครนอก", "ถนนนครใน"]:
            if st in combined_text:
                matched_street = st
                break

        # Dynamic Footer Buttons & Contextual Quick Replies
        footer_buttons = []
        custom_quick_replies = None

        if matched_place:
            pname = matched_place.get("name", "สถานที่").split("(")[0].strip()
            maps_url = SongkhlaFlexTemplates.clean_maps_url(matched_place.get("google_maps_url"))
            tel_uri = SongkhlaFlexTemplates.clean_tel_uri(matched_place.get("phone", ""), default="")
            
            # Button 1: Google Maps Navigation
            footer_buttons.append({
                "type": "button",
                "style": "primary",
                "color": "#0984e3",
                "height": "sm",
                "action": {
                    "type": "uri",
                    "label": f"📍 แผนที่ {pname[:8]}",
                    "uri": maps_url
                }
            })
            
            # Button 2: Direct Call or Related Food
            if tel_uri:
                footer_buttons.append({
                    "type": "button",
                    "style": "secondary",
                    "height": "sm",
                    "action": {
                        "type": "uri",
                        "label": "📞 โทรติดต่อ",
                        "uri": tel_uri
                    }
                })
            else:
                footer_buttons.append({
                    "type": "button",
                    "style": "secondary",
                    "height": "sm",
                    "action": {
                        "type": "message",
                        "label": "🍜 ร้านอาหารเด็ด",
                        "text": "สำรวจร้านอาหารทั้งหมด"
                    }
                })
            
            custom_quick_replies = [
                ("🍴 เมนูเด่นร้านนี้", f"เมนูเด่นของ {pname} มีอะไรบ้าง"),
                ("🕒 เวลาเปิด-ปิด", f"เวลาเปิดปิดของ {pname}"),
                ("🚶 ที่เที่ยวใกล้เคียง", f"สถานที่ใกล้เคียง {pname} มีอะไรบ้าง"),
                ("🧭 หน้าแรกนำทาง", "หน้าแรก")
            ]

        elif matched_street:
            footer_buttons.append({
                "type": "button",
                "style": "primary",
                "color": "#0984e3",
                "height": "sm",
                "action": {
                    "type": "message",
                    "label": f"🍜 ร้านบน{matched_street[:8]}",
                    "text": f"มีร้านอาหารอะไรบ้างบน{matched_street}"
                }
            })
            footer_buttons.append({
                "type": "button",
                "style": "secondary",
                "height": "sm",
                "action": {
                    "type": "message",
                    "label": f"📸 ที่เที่ยวบน{matched_street[:8]}",
                    "text": f"มีสถานที่ท่องเที่ยวอะไรบ้างบน{matched_street}"
                }
            })
            custom_quick_replies = [
                (f"🍜 ของกินบน{matched_street[:6]}", f"ร้านอาหารบน{matched_street}"),
                (f"📸 จุดเช็คอินบน{matched_street[:6]}", f"จุดถ่ายรูปบน{matched_street}"),
                ("🧭 หน้าแรกนำทาง", "หน้าแรก")
            ]
        else:
            footer_buttons.append({
                "type": "button",
                "style": "primary",
                "color": "#0984e3",
                "height": "sm",
                "action": {
                    "type": "message",
                    "label": "🧭 หน้าแรกนำทาง",
                    "text": "หน้าแรก"
                }
            })
            footer_buttons.append({
                "type": "button",
                "style": "secondary",
                "height": "sm",
                "action": {
                    "type": "message",
                    "label": "🍜 สำรวจร้านอาหาร",
                    "text": "สำรวจร้านอาหารทั้งหมด"
                }
            })

        bubble = {
            "type": "bubble",
            "header": {
                "type": "box",
                "layout": "vertical",
                "contents": [
                    {
                        "type": "box",
                        "layout": "horizontal",
                        "contents": [
                            {
                                "type": "box",
                                "layout": "horizontal",
                                "spacing": "xs",
                                "contents": [
                                    {"type": "text", "text": "🏮", "size": "sm", "flex": 0},
                                    {"type": "text", "text": "น้องสิงขร ตอบคำถาม", "weight": "bold", "size": "sm", "color": "#1e272e"}
                                ]
                            },
                            {
                                "type": "box",
                                "layout": "horizontal",
                                "backgroundColor": badge_bg,
                                "cornerRadius": "sm",
                                "paddingStart": "sm",
                                "paddingEnd": "sm",
                                "paddingTop": "xs",
                                "paddingBottom": "xs",
                                "contents": [
                                    {
                                        "type": "text",
                                        "text": badge_text,
                                        "size": "xxs",
                                        "color": badge_color,
                                        "weight": "bold"
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "type": "separator",
                        "margin": "md",
                        "color": "#ecf0f1"
                    }
                ]
            },
            "body": {
                "type": "box",
                "layout": "vertical",
                "contents": [
                    {
                        "type": "text",
                        "text": clean_answer,
                        "size": "sm",
                        "color": "#2c3e50",
                        "wrap": True,
                        "lineSpacing": "5px"
                    },
                    {
                        "type": "box",
                        "layout": "vertical",
                        "backgroundColor": "#f8f9fa",
                        "cornerRadius": "md",
                        "paddingAll": "sm",
                        "margin": "lg",
                        "spacing": "xs",
                        "contents": [
                            {
                                "type": "text",
                                "text": "📖 ข้อมูลอ้างอิงที่ตรวจสอบแล้ว:",
                                "size": "xxs",
                                "weight": "bold",
                                "color": "#7f8c8d"
                            },
                            *source_texts
                        ]
                    }
                ]
            },
            "footer": {
                "type": "box",
                "layout": "horizontal",
                "spacing": "sm",
                "contents": footer_buttons
            }
        }
        return FlexSendMessage(
            alt_text=f"คำตอบจากน้องสิงขร: {clean_answer[:35]}...",
            contents=bubble,
            quick_reply=SongkhlaFlexTemplates.get_quick_replies(custom_quick_replies)
        )
