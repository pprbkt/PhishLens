"""
Dataset downloader and synthesizer script for PhishLens.
Fetches the UCI SMS Spam dataset and enriches it with modern smishing and benign examples.
"""

import csv
import logging
import os
import urllib.request
from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
OUTPUT_CSV = os.path.join(RAW_DATA_DIR, "sms_dataset.csv")

UCI_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

# Modern curated smishing and benign examples to enrich model domain knowledge
MODERN_SAMPLES: List[Tuple[str, str]] = [
    # Modern Phishing / Smishing
    ("URGENT: Your SBI account has been blocked due to incomplete KYC. Update your PAN card immediately at http://sbi-kyc-update.top or service will stop.", "scam"),
    ("Dear customer, your electricity power will be disconnected tonight at 9:30 PM due to unpaid bill. Call electricity officer at 9876543210 immediately.", "scam"),
    ("Your FedEx package #US98421 could not be delivered due to incomplete address. Please update details and pay $1.99 redelivery fee at http://fedx-tracking-portal.com", "scam"),
    ("USPS: We tried to deliver your parcel today but no one was home. Confirm your address at http://usps-redelivery-info.site within 24 hours.", "scam"),
    ("HDFC Bank Alert: ₹85,000 debited from your A/C via UPI to amazon@paytm. If not you, block transaction immediately: http://hdfc-fraud-block.net/cancel", "scam"),
    ("Your Netflix subscription has expired. Update your billing information at http://netflix-billing-renew.xyz to avoid suspension of streaming service.", "scam"),
    ("Apple Security: Your iCloud ID was accessed from an unrecognized device in Moscow, Russia. If not you, lock account at http://appleid-security-verify.co", "scam"),
    ("PayPal Alert: You sent a payment of $499.00 USD to Coinbase Inc. If this was not authorized, dispute charge at http://paypal-resolution-center.biz", "scam"),
    ("Dear User, you have won ₹1,50,000 cash prize in Kaun Banega Crorepati WhatsApp Lucky Draw! Send details to +919876543210 on WhatsApp to claim prize.", "scam"),
    ("Income Tax Dept: Tax refund of ₹24,800 has been approved. Enter bank details to receive refund in 2 hours: http://incometax-refund-gov.in.top", "scam"),
    ("Binance Security Alert: New withdrawal request of 0.85 BTC initiated. If this wasn't you, cancel immediately at http://binance-security-cancel.org", "scam"),
    ("Amazon: Your order #402-984218 for iPhone 15 Pro ($1,199.00) is confirmed. If you did not place this, call fraud prevention immediately at 1-800-456-7890.", "scam"),
    ("Alert: Your SIM card will be deactivated within 24 hours due to non-verification. Click http://airtel-sim-kyc.pw to submit Aadhaar details.", "scam"),
    ("Chase Alert: Suspicious sign-in detected from Chrome on Linux. Verify identity now at http://chase-online-protect.top to secure funds.", "scam"),
    ("Congratulations! You are pre-approved for an instant loan of ₹5,00,000 with 0% interest for 1 year. Apply now at http://instant-credit-loan.xyz/apply", "scam"),
    ("Geek Squad: Auto-renewal of Geek Protection plan ($399.99) charged to your card. Call +1-888-999-0123 within 24 hours for full refund.", "scam"),
    ("Wells Fargo: Your debit card is restricted due to unusual activity. Reactivate your card at http://wellsfargo-debit-unlock.cc immediately.", "scam"),
    ("Toll Enforcement: You have an unpaid toll invoice of $12.50. Avoid $50 penalty by paying online today at http://tollservices-pay-online.info", "scam"),
    ("Meta Support: Your Instagram account will be deleted in 24 hours due to copyright violation. Appeal here: http://instagram-copyright-appeal.com", "scam"),
    ("Crypto Giveaway: Elon Musk is giving away 5,000 ETH! Send 0.1 ETH to verify wallet and receive 1.0 ETH back instantly at http://tesla-eth-promo.org", "scam"),

    # Modern Legitimate / Benign (Transactional, OTPs, Chat)
    ("Your one-time password (OTP) for logging into your HDFC Bank account is 482910. Do not share this OTP with anyone, including bank officials.", "not_scam"),
    ("Uber: Your driver Rajesh (Swift Dzire - MH02AB1234) is arriving in 3 mins. OTP for this ride is 7731.", "not_scam"),
    ("Swiggy: Your order from Burger King has been picked up by delivery partner Amit. Expected delivery by 8:45 PM.", "not_scam"),
    ("Zomato: Order confirmed! Pizza Hut is preparing your meal. Track your delivery live in the app.", "not_scam"),
    ("Your appointment with Dr. Sharma is confirmed for tomorrow at 4:30 PM at Apollo Clinic. Please arrive 10 minutes early.", "not_scam"),
    ("Amazon Delivery Update: Your package containing 'Wireless Bluetooth Headphones' is out for delivery today with associate Ramesh.", "not_scam"),
    ("Your monthly electricity bill of ₹1,420 for consumer #98214 is due on 28th Sep. Pay via official mobile app or electricity board website.", "not_scam"),
    ("Aadhaar OTP for authentication is 918234. Valid for 10 minutes. UIDAI will never call asking for OTP.", "not_scam"),
    ("Hi Mom, I reached the hotel safely. Will call you once I finish dinner.", "not_scam"),
    ("Hey, are we still meeting for coffee at Starbucks around 5 PM today?", "not_scam"),
    ("Your salary of ₹75,000 has been credited to A/C XX4921 on 24-Sep-2026. Avl Bal: ₹89,450. - State Bank of India", "not_scam"),
    ("Google: 593021 is your 2-step verification code. Never share this code with anyone.", "not_scam"),
    ("Your flight 6E-204 from Delhi to Mumbai is on schedule. Web check-in closes 60 minutes before departure.", "not_scam"),
    ("Can you please send me the presentation slides before the 3 PM team meeting?", "not_scam"),
    ("Reminder: Library books 'Clean Code' and 'Design Patterns' are due for return by Friday, 26th Sep.", "not_scam")
]


def download_and_prepare_dataset() -> str:
    """Download base SMS dataset and merge with modern phishing dataset."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    rows: List[Tuple[str, str]] = []

    # 1. Attempt to fetch benchmark dataset
    try:
        logger.info("Fetching UCI SMS Spam Collection from %s...", UCI_URL)
        req = urllib.request.Request(UCI_URL, headers={"User-Agent": "PhishLens-Downloader/1.0"})
        with urllib.request.urlopen(req, timeout=15) as response:
            content = response.read().decode("utf-8", errors="ignore")
            lines = content.strip().split("\n")
            for line in lines:
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    raw_label, text = parts
                    raw_label = raw_label.strip().lower()
                    label = "scam" if raw_label == "spam" else "not_scam"
                    text = text.strip()
                    if text:
                        rows.append((text, label))
            logger.info("Successfully fetched %d records from UCI SMS repository.", len(rows))
    except Exception as e:
        logger.warning("Could not download UCI dataset: %s. Using local fallback dataset.", e)

    # 2. Add modern curated examples
    existing_texts = {r[0] for r in rows}
    added_modern = 0
    for text, label in MODERN_SAMPLES:
        if text not in existing_texts:
            rows.append((text, label))
            existing_texts.add(text)
            added_modern += 1

    logger.info("Added %d modern phishing and benign samples.", added_modern)

    # 3. Write to unified CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        for text, label in rows:
            writer.writerow([text, label])

    logger.info("Saved unified raw dataset (%d records) to %s", len(rows), OUTPUT_CSV)
    return OUTPUT_CSV


if __name__ == "__main__":
    download_and_prepare_dataset()
