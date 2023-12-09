from flask import logging
from flask_mail import Mail, Message
from app.engine import MODEL_REGISTRY, Pipeline2, OptimizerFactory, ModelBuilder
from datetime import datetime
from app import create_app, db, celery, mail, socketio
from app.dbdata import TaskResult



def convert_to_date_only(datetime_obj):
    if isinstance(datetime_obj, datetime):
        return datetime_obj.strftime("%Y-%m-%d")
    elif isinstance(datetime_obj, str):
        dt_obj = datetime.strptime(datetime_obj, "%Y-%m-%dT%H:%M:%S.%fZ")
        return dt_obj.strftime("%Y-%m-%d")
    else:
        raise TypeError("Unsupported type for datetime conversion")

@celery.task
def send_activation_email(email, activation_link):
    with create_app().app_context():
        msg = Message('Account Activation', recipients=[email])
        msg.body = f'Click the following link to activate your account: {activation_link}'
        mail.send(msg)

@celery.task
def send_password_reset_email(email, reset_link):
    with create_app().app_context():
        msg = Message('Password Reset', recipients=[email])
        msg.body = f'Click the following link to reset your password: {reset_link}'
        mail.send(msg)

@celery.task(name='app.train_model_task', bind=True)
def train_model_task(self, configurations):
    with create_app().app_context():
        try:
            all_results = []
            print(f"Processing {len(configurations)} configurations...")

            for config in configurations:
                print(f"Processing configuration for symbol: {config['symbol']}")
                symbol = config['symbol']
                start_date = datetime.strptime(config['start_date'], "%Y-%m-%dT%H:%M:%S.%fZ")
                end_date = datetime.strptime(config['end_date'], "%Y-%m-%dT%H:%M:%S.%fZ")
                start_date_str = convert_to_date_only(start_date)
                end_date_str = convert_to_date_only(end_date)

                model_builders = []

                for model_name in config['model_name']:
                    model_config = MODEL_REGISTRY.get(model_name)
                    if model_config:  # Ensure model_config is not None
                        model_class = model_config['class']
                        param_grid = model_config['param_grid']
                        optimizer_type = model_config['optimizer_type']
                        if optimizer_type:
                            param_grid = model_config.get('param_grid', {})  # Fetch the param_grid, provide a default if not present
                            optimizer = OptimizerFactory.create_optimizer(optimizer_type, param_grid=param_grid)
                        else:
                            optimizer = None  
                        
                        model_builder = ModelBuilder(model_class, optimizer)
                        model_builders.append(model_builder)

                stocks_info = [(symbol, start_date_str, end_date_str, 14, 3, model_builders)]
                pipeline = Pipeline2(stocks_info)
                predictions = pipeline.process_data_pipeline()
                print(f"Predictions for {symbol}: {predictions}")  # Debugging log

                # Collecting metrics for each prediction
                for prediction in predictions:
                    si, test_predictions, mae_val, mae_test, mse_val, mse_test, r2_val, r2_test, mape_val, mape_test = prediction
                    all_results.append({
                        'symbol': symbol, 
                        'mae_val': mae_val, 
                        'mae_test': mae_test,
                        'mse_val': mse_val,
                        'mse_test': mse_test,
                        'r2_val': r2_val,
                        'r2_test': r2_test,
                        'mape_val': mape_val,
                        'mape_test': mape_test
                    })

            # Storing results in database and emitting completion message
            result = TaskResult(task_id=self.request.id, result=all_results)
            db.session.add(result)
            db.session.commit()

            socketio.emit('training_complete', {'progress': 100, 'message': 'Training completed!', 'predictions': all_results})

            return all_results

        except Exception as e:
            print(f"Error occurred: {e}")  # Debugging log
            db.session.rollback()
            raise e


def convert_to_date_only(datetime_obj):
    """Converts datetime object to date string."""
    return datetime_obj.strftime("%Y-%m-%d")
