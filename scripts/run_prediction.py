#!/usr/bin/env python
"""
Prediction script for the car price prediction project.

This script loads a trained model and makes predictions on new data.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

from src.models.predict import ModelPredictor
from src.utils.helpers import setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Make car price predictions"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained model"
    )
    parser.add_argument(
        "--preprocessor-path",
        type=str,
        default=None,
        help="Path to preprocessor"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input CSV file for batch prediction"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save predictions"
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Make a single prediction (interactive mode)"
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="JSON string with car details for prediction"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    return parser.parse_args()


def get_user_input():
    """Get car details from user input."""
    print("\n" + "="*50)
    print("Enter Car Details for Price Prediction")
    print("="*50 + "\n")
    
    try:
        year = int(input("Year of manufacture (e.g., 2018): "))
        km_driven = int(input("Kilometers driven (e.g., 50000): "))
        fuel = input("Fuel type (Petrol/Diesel/CNG/LPG/Electric): ")
        seller_type = input("Seller type (Individual/Dealer/Trustmark Dealer): ")
        transmission = input("Transmission (Manual/Automatic): ")
        owner = input("Owner (First Owner/Second Owner/Third Owner/Fourth & Above Owner): ")
        mileage = float(input("Mileage in km/l (e.g., 18.5): "))
        engine = float(input("Engine capacity in CC (e.g., 1500): "))
        max_power = float(input("Max power in bhp (e.g., 120): "))
        seats = int(input("Number of seats (e.g., 5): "))
        
        return {
            'year': year,
            'km_driven': km_driven,
            'fuel': fuel,
            'seller_type': seller_type,
            'transmission': transmission,
            'owner': owner,
            'mileage': mileage,
            'engine': engine,
            'max_power': max_power,
            'seats': seats
        }
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        return None


def main():
    """Main prediction function."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level)
    
    logger.info("="*60)
    logger.info("Car Price Prediction - Prediction Module")
    logger.info("="*60)
    
    try:
        # Initialize predictor
        model_path = args.model_path
        if model_path is None:
            model_path = project_root / "models" / "best_model.pkl"
        
        preprocessor_path = args.preprocessor_path
        if preprocessor_path is None:
            preprocessor_path = project_root / "models" / "preprocessor.pkl"
        
        predictor = ModelPredictor(
            model_path=model_path,
            preprocessor_path=preprocessor_path
        )
        
        # Batch prediction from file
        if args.input:
            logger.info(f"Making batch predictions from {args.input}")
            
            output_path = args.output or args.input.replace('.csv', '_predictions.csv')
            
            results_df = predictor.batch_predict(
                filepath=args.input,
                output_path=output_path
            )
            
            print("\n" + "="*50)
            print("Prediction Results Summary")
            print("="*50)
            print(f"Total records: {len(results_df)}")
            print(f"Average predicted price: {results_df['predicted_price'].mean():,.2f}")
            print(f"Min predicted price: {results_df['predicted_price'].min():,.2f}")
            print(f"Max predicted price: {results_df['predicted_price'].max():,.2f}")
            print(f"\nResults saved to: {output_path}")
            
        # Single prediction from JSON
        elif args.json:
            car_data = json.loads(args.json)
            logger.info(f"Making prediction for: {car_data}")
            
            prediction = predictor.predict(car_data)
            
            print("\n" + "="*50)
            print("Prediction Result")
            print("="*50)
            print(f"Predicted Price: ₹{prediction[0]:,.2f}")
            
        # Interactive single prediction
        elif args.single:
            car_data = get_user_input()
            
            if car_data:
                logger.info("Making prediction...")
                
                prediction = predictor.predict_single(**car_data)
                
                print("\n" + "="*50)
                print("Prediction Result")
                print("="*50)
                print(f"\nPredicted Selling Price: ₹{prediction:,.2f}")
                print("\n" + "="*50)
                
                # Try to get confidence interval
                try:
                    confidence = predictor.predict_with_confidence(car_data)
                    if confidence['std'][0] > 0:
                        print(f"95% Confidence Interval: ₹{confidence['lower_95'][0]:,.2f} - ₹{confidence['upper_95'][0]:,.2f}")
                except:
                    pass
        
        else:
            # Default: show help
            print("\nUsage examples:")
            print("\n1. Batch prediction from file:")
            print("   python run_prediction.py --input data/new_cars.csv --output predictions.csv")
            print("\n2. Single prediction (interactive):")
            print("   python run_prediction.py --single")
            print("\n3. Single prediction from JSON:")
            print('   python run_prediction.py --json \'{"year": 2018, "km_driven": 50000, ...}\'')
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.info("Make sure to run the training script first: python scripts/run_training.py")
        return 1
        
    except Exception as e:
        logger.exception(f"Prediction failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())